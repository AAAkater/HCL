#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>
#include <stdexcept>

namespace hcl
{
inline bool nativeHclBatchUsesWrapCapture(uint64_t currentTarget, size_t rangeNode)
{
    // Batch replay advances the completion target before the next range starts.
    // A suffix uses that new target, not the original batch's graph offset.
    return (currentTarget % 64 + rangeNode % 64) % 64 == 0;
}

enum class NativeHclRelocationKind : uint8_t
{
    DEPENDENCY_TARGET,
    DEPENDENCY_LONG_SO_INDEX,
    EXTERNAL_COMPLETION_TARGET,
    INTERNAL_COMPLETION_TARGET,
    EXTERNAL_COMPLETION_ADDRESS,
    INTERNAL_COMPLETION_ADDRESS,
    GPSO_0_ADDRESS,
    GPSO_1_ADDRESS,
    RECEIVE_STATE,
    QUEUE_COUNTER,
    WQE_COUNTER,
    WQE_NOTIFY,
    WQE_WAIT,
    EXTERNAL_COMPLETION_INDEX,
    INTERNAL_COMPLETION_INDEX,
    GPSO_0_INDEX,
    GPSO_1_INDEX,
    PENDING_COMPLETION_ADDRESS,
    PENDING_COMPLETION_INDEX,
    EXTERNAL_SHORT_MONITOR,
    INTERNAL_SHORT_MONITOR,
    GPSO_0_SHORT_MONITOR,
    GPSO_1_SHORT_MONITOR,
    PENDING_SHORT_MONITOR,
    COUNT
};

// Short monitors select one SOB within an eight-object group. The monitor's
// configured quarter is fixed for the retained command; only SID/mask move.
inline uint32_t nativeHclShortMonitorIndex(uint32_t arm)
{
    const uint32_t selected = (~(arm >> 8)) & 0xffu;
    if (selected == 0 || (selected & (selected - 1)) != 0)
        throw std::invalid_argument("Native short monitor must track exactly one SOB");
    unsigned lane = 0;
    while ((selected & (1u << lane)) == 0) ++lane;
    return (arm & 0xffu) * 8 + lane;
}

inline uint32_t nativeHclShortMonitorBits(uint32_t capturedIndex,
                                        uint32_t capturedAddress,
                                        uint32_t currentAddress)
{
    const int64_t delta = int64_t(currentAddress) - int64_t(capturedAddress);
    const int64_t index = int64_t(capturedIndex) + delta / 4;
    if (capturedIndex >= 2048 || delta % 4 != 0 || index < 0 || index >= 2048)
        throw std::invalid_argument("Native short monitor moved outside its captured quarter");
    return (uint32_t(index) >> 3) | ((~(1u << (index & 7)) & 0xffu) << 8);
}

struct NativeHclRelocation
{
    uint32_t                commandIndex    = 0;
    uint32_t                byteOffset      = 0;
    uint8_t                 width           = 0;
    uint64_t                mask            = 0;
    uint8_t                 destinationShift = 0;
    uint8_t                 sourceShift      = 0;
    NativeHclRelocationKind kind             = NativeHclRelocationKind::DEPENDENCY_TARGET;
    uint32_t                capturedAddress  = 0;
};

// Capture a packed firmware bitfield without guessing its ABI bit offsets.
// The original command is restored before returning; replay patches only
// these field bits, leaving adjacent flags and reduction settings untouched.
template<class Setter>
std::vector<NativeHclRelocation> nativeHclBitfieldSites(
    void* packet, size_t size, uint32_t capturedValue, Setter&& setter,
    NativeHclRelocationKind kind, uint32_t capturedAddress)
{
    auto* bytes = static_cast<uint8_t*>(packet);
    setter(0);
    const std::vector<uint8_t> zero(bytes, bytes + size);
    setter(UINT32_MAX);
    const std::vector<uint8_t> ones(bytes, bytes + size);
    setter(capturedValue);
    std::vector<NativeHclRelocation> result;
    uint8_t sourceShift = 0;
    for (size_t index = 0; index < size; ++index)
    {
        const uint8_t mask = zero[index] ^ ones[index];
        if (!mask) continue;
        uint8_t destinationShift = 0;
        while (!(mask & (1u << destinationShift))) ++destinationShift;
        const unsigned field = mask >> destinationShift;
        if ((field & (field + 1)) != 0) throw std::invalid_argument("Noncontiguous native bitfield");
        uint8_t bits = 0;
        for (unsigned value = field; value; value >>= 1) ++bits;
        result.push_back({0, static_cast<uint32_t>(index), 1, mask, destinationShift, sourceShift,
                          kind, capturedAddress});
        sourceShift += bits;
    }
    if (!sourceShift || sourceShift > 32) throw std::invalid_argument("Invalid native bitfield width");
    return result;
}

struct NativeHclCommand
{
    std::vector<uint8_t> bytes;
    bool                 submitAfter = false;
};

struct NativeHclStreamTemplate
{
    std::vector<NativeHclCommand>    commands;
    std::vector<NativeHclRelocation> relocations;
    uint64_t                         captureTargetValue = 0;

    bool empty() const { return commands.empty(); }
};

struct NativeHclReplayState
{
    uint64_t values[static_cast<size_t>(NativeHclRelocationKind::COUNT)] {};

    uint64_t get(NativeHclRelocationKind kind) const
    {
        return values[static_cast<size_t>(kind)];
    }
};

inline bool isNativeHclRelocationWellFormed(const NativeHclRelocation& relocation)
{
    if (relocation.width != 1 && relocation.width != 2 && relocation.width != 4 && relocation.width != 8)
    {
        return false;
    }
    if (relocation.destinationShift >= relocation.width * 8 || relocation.sourceShift >= 64 ||
        relocation.kind >= NativeHclRelocationKind::COUNT)
    {
        return false;
    }
    const uint64_t widthMask = relocation.width == 8
                                   ? std::numeric_limits<uint64_t>::max()
                                   : ((uint64_t {1} << (relocation.width * 8)) - 1);
    if (relocation.mask == 0 || (relocation.mask & ~widthMask) != 0)
    {
        return false;
    }
    const uint64_t bitsBelowDestination = relocation.destinationShift == 0
                                              ? 0
                                              : ((uint64_t {1} << relocation.destinationShift) - 1);
    return (relocation.mask & bitsBelowDestination) == 0;
}

inline bool nativeHclRelocationLess(const NativeHclRelocation& left, const NativeHclRelocation& right)
{
    if (left.commandIndex != right.commandIndex)
    {
        return left.commandIndex < right.commandIndex;
    }
    if (left.byteOffset != right.byteOffset)
    {
        return left.byteOffset < right.byteOffset;
    }
    return left.mask < right.mask;
}

inline bool validateNativeHclStreamTemplate(const NativeHclStreamTemplate& commandTemplate)
{
    if (commandTemplate.empty())
    {
        return false;
    }
    for (size_t index = 0; index < commandTemplate.relocations.size(); ++index)
    {
        const NativeHclRelocation& relocation = commandTemplate.relocations[index];
        if (!isNativeHclRelocationWellFormed(relocation) ||
            relocation.commandIndex >= commandTemplate.commands.size())
        {
            return false;
        }
        const NativeHclCommand& command = commandTemplate.commands[relocation.commandIndex];
        if (relocation.byteOffset > command.bytes.size() ||
            relocation.width > command.bytes.size() - relocation.byteOffset)
        {
            return false;
        }
        if (index != 0 && nativeHclRelocationLess(relocation, commandTemplate.relocations[index - 1]))
        {
            return false;
        }
    }
    return true;
}

// Completion SOB addresses advance with the HCL completion epoch. The graph
// generators identify the semantic address at capture time, while scheduler
// packet writers flatten it into one or more 32-bit LBW fields. Add the
// corresponding fixed-address relocations once during instantiation.
inline size_t addNativeHclCapturedWordRelocations(NativeHclStreamTemplate& commandTemplate,
                                                  uint32_t                 capturedWord,
                                                  NativeHclRelocationKind  kind)
{
    size_t added = 0;
    for (size_t commandIndex = 0; commandIndex < commandTemplate.commands.size(); ++commandIndex)
    {
        const NativeHclCommand& command = commandTemplate.commands[commandIndex];
        for (size_t byteOffset = 0; byteOffset + sizeof(uint32_t) <= command.bytes.size();
             byteOffset += sizeof(uint32_t))
        {
            uint32_t word = 0;
            std::memcpy(&word, command.bytes.data() + byteOffset, sizeof(word));
            if (word != capturedWord)
            {
                continue;
            }
            commandTemplate.relocations.push_back({.commandIndex = static_cast<uint32_t>(commandIndex),
                                                    .byteOffset = static_cast<uint32_t>(byteOffset),
                                                    .width = sizeof(uint32_t),
                                                    .mask = 0xffffffffu,
                                                    .destinationShift = 0,
                                                    .sourceShift = 0,
                                                    .kind = kind});
            ++added;
        }
    }
    std::sort(commandTemplate.relocations.begin(),
              commandTemplate.relocations.end(),
              nativeHclRelocationLess);
    return added;
}

inline bool patchNativeHclRelocation(uint8_t*                    commandBytes,
                                     size_t                      commandSize,
                                     const NativeHclRelocation&  relocation,
                                     const NativeHclReplayState& replayState)
{
    if (commandBytes == nullptr || !isNativeHclRelocationWellFormed(relocation) ||
        relocation.byteOffset > commandSize || relocation.width > commandSize - relocation.byteOffset)
    {
        return false;
    }

    uint64_t word = 0;
    std::memcpy(&word, commandBytes + relocation.byteOffset, relocation.width);
    const uint64_t value = replayState.get(relocation.kind) >> relocation.sourceShift;
    word = (word & ~relocation.mask) | ((value << relocation.destinationShift) & relocation.mask);
    std::memcpy(commandBytes + relocation.byteOffset, &word, relocation.width);
    return true;
}
}  // namespace hcl
