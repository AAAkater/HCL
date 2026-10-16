#include "infra/scal/gen2_arch_common/native_hcl_relocation.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

uint32_t readWord(const std::array<uint8_t, 16>& bytes, size_t offset)
{
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    return word;
}

void testLongSoChunks(uint64_t target)
{
    constexpr uint32_t preserved = 0x00015500u;
    std::array<uint8_t, 16> bytes {};
    for (size_t chunk = 0; chunk < 4; ++chunk)
    {
        std::memcpy(bytes.data() + chunk * sizeof(uint32_t), &preserved, sizeof(preserved));
    }

    hcl::NativeHclReplayState state;
    state.values[static_cast<size_t>(hcl::NativeHclRelocationKind::DEPENDENCY_TARGET)] = target;
    for (size_t chunk = 0; chunk < 4; ++chunk)
    {
        hcl::NativeHclRelocation relocation;
        relocation.byteOffset      = static_cast<uint32_t>(chunk * sizeof(uint32_t));
        relocation.width           = sizeof(uint32_t);
        relocation.mask            = 0xfffe0000u;
        relocation.destinationShift = 17;
        relocation.sourceShift      = static_cast<uint8_t>(chunk * 15);
        relocation.kind             = hcl::NativeHclRelocationKind::DEPENDENCY_TARGET;
        require(hcl::patchNativeHclRelocation(bytes.data(), bytes.size(), relocation, state),
                "long-SO relocation rejected");
        const uint32_t expectedChunk = static_cast<uint32_t>((target >> (chunk * 15)) & 0x7fffu);
        const uint32_t actual = readWord(bytes, chunk * sizeof(uint32_t));
        require((actual >> 17) == expectedChunk, "long-SO chunk mismatch");
        require((actual & ~0xfffe0000u) == preserved, "long-SO relocation changed unrelated bits");
    }
}

void testSegmentedBatchPhases()
{
    for (uint64_t phase = 0; phase < 64; ++phase)
        for (size_t count = 2; count <= 128; ++count)
            for (size_t split = 1; split < count; ++split)
                for (size_t node = 0; node < count; ++node)
                {
                    const uint64_t base = 64 + phase;
                    const bool expected = (base + node) % 64 == 0;
                    const bool actual = node < split
                        ? hcl::nativeHclBatchUsesWrapCapture(base, node)
                        : hcl::nativeHclBatchUsesWrapCapture(base + split, node - split);
                    require(actual == expected, "segmented completion-ring phase differs from whole replay");
                }
    require(hcl::nativeHclBatchUsesWrapCapture(64, 0), "suffix must wrap at its current completion target");
    require(!hcl::nativeHclBatchUsesWrapCapture(64, 3), "global graph offset must not be added twice");
}

void testWqeBits()
{
    std::array<uint8_t, 1> bytes {0b10100101u};
    hcl::NativeHclReplayState state;
    hcl::NativeHclRelocation notify;
    notify.width = 1;
    notify.mask = 0x08;
    notify.destinationShift = 3;
    notify.kind = hcl::NativeHclRelocationKind::WQE_NOTIFY;
    hcl::NativeHclRelocation wait = notify;
    wait.mask = 0x40;
    wait.destinationShift = 6;
    wait.kind = hcl::NativeHclRelocationKind::WQE_WAIT;

    state.values[static_cast<size_t>(hcl::NativeHclRelocationKind::WQE_NOTIFY)] = 1;
    state.values[static_cast<size_t>(hcl::NativeHclRelocationKind::WQE_WAIT)] = 0;
    require(hcl::patchNativeHclRelocation(bytes.data(), bytes.size(), notify, state), "WQE notify patch rejected");
    require(hcl::patchNativeHclRelocation(bytes.data(), bytes.size(), wait, state), "WQE wait patch rejected");
    require(bytes[0] == 0b10101101u, "WQE patch changed the wrong bits");

    state.values[static_cast<size_t>(hcl::NativeHclRelocationKind::WQE_NOTIFY)] = 0;
    state.values[static_cast<size_t>(hcl::NativeHclRelocationKind::WQE_WAIT)] = 1;
    require(hcl::patchNativeHclRelocation(bytes.data(), bytes.size(), notify, state), "WQE notify clear rejected");
    require(hcl::patchNativeHclRelocation(bytes.data(), bytes.size(), wait, state), "WQE wait set rejected");
    require(bytes[0] == 0b11100101u, "WQE wrap patch changed the wrong bits");
}

void testTemplateValidation()
{
    hcl::NativeHclStreamTemplate commandTemplate;
    commandTemplate.commands.resize(2);
    commandTemplate.commands[0].bytes.resize(8);
    commandTemplate.commands[1].bytes.resize(8);

    hcl::NativeHclRelocation first;
    first.commandIndex = 0;
    first.byteOffset = 4;
    first.width = 4;
    first.mask = 0xfffe0000u;
    first.destinationShift = 17;
    hcl::NativeHclRelocation second = first;
    second.commandIndex = 1;
    second.byteOffset = 0;
    commandTemplate.relocations = {first, second};
    require(hcl::validateNativeHclStreamTemplate(commandTemplate), "valid relocation template rejected");

    std::swap(commandTemplate.relocations[0], commandTemplate.relocations[1]);
    require(!hcl::validateNativeHclStreamTemplate(commandTemplate), "unsorted relocation template accepted");
    std::swap(commandTemplate.relocations[0], commandTemplate.relocations[1]);
    commandTemplate.relocations[1].byteOffset = 6;
    require(!hcl::validateNativeHclStreamTemplate(commandTemplate), "out-of-range relocation accepted");
    commandTemplate.relocations[1].byteOffset = 0;
    commandTemplate.relocations[1].width = 3;
    require(!hcl::validateNativeHclStreamTemplate(commandTemplate), "invalid relocation width accepted");
}

void testCapturedCompletionAddress()
{
    constexpr uint32_t capturedAddress = 0xf3a01234u;
    constexpr uint32_t replayAddress = 0xf3a01238u;
    hcl::NativeHclStreamTemplate commandTemplate;
    commandTemplate.commands.resize(2);
    commandTemplate.commands[0].bytes.resize(12);
    commandTemplate.commands[1].bytes.resize(9);
    std::memcpy(commandTemplate.commands[0].bytes.data() + 4, &capturedAddress, sizeof(capturedAddress));
    // An unaligned copy must not be treated as a scheduler dword field.
    std::memcpy(commandTemplate.commands[1].bytes.data() + 1, &capturedAddress, sizeof(capturedAddress));

    const size_t added = hcl::addNativeHclCapturedWordRelocations(
        commandTemplate,
        capturedAddress,
        hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS);
    require(added == 1, "completion-address discovery did not select exactly the aligned field");
    require(hcl::validateNativeHclStreamTemplate(commandTemplate),
            "completion-address relocation made the template invalid");

    hcl::NativeHclReplayState state;
    state.values[static_cast<size_t>(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS)] = replayAddress;
    const auto& relocation = commandTemplate.relocations.front();
    require(hcl::patchNativeHclRelocation(commandTemplate.commands[0].bytes.data(),
                                          commandTemplate.commands[0].bytes.size(),
                                          relocation,
                                          state),
            "completion-address relocation rejected");
    uint32_t actual = 0;
    std::memcpy(&actual, commandTemplate.commands[0].bytes.data() + 4, sizeof(actual));
    require(actual == replayAddress, "completion-address relocation patched the wrong value");
}
void testEncodedNicSobAddress()
{
    constexpr uint32_t addressMask = 0x07ffffffu;
    hcl::NativeHclRelocation relocation;
    relocation.width = sizeof(uint32_t);
    relocation.mask = addressMask;
    relocation.kind = hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS;
    for (const uint32_t control : {0x48000000u, 0x68000000u})
    {
        std::array<uint8_t, 16> bytes {};
        uint32_t encoded = control | 0x00123400u;
        std::memcpy(bytes.data(), &encoded, sizeof(encoded));
        // Cross an address carry and revisit the first slot after a full ring.
        for (uint32_t slot = 0; slot != 2049; ++slot)
        {
            const uint32_t address = 0xf8000000u | (0x000ff800u + (slot % 1024) * 4);
            hcl::NativeHclReplayState state;
            state.values[static_cast<size_t>(relocation.kind)] = address;
            require(hcl::patchNativeHclRelocation(bytes.data(), bytes.size(), relocation, state),
                    "encoded NIC SOB relocation rejected");
            const uint32_t actual = readWord(bytes, 0);
            require((actual & addressMask) == (address & addressMask), "NIC SOB address did not follow replay epoch");
            require((actual & ~addressMask) == control, "NIC SOB relocation changed completion control bits");
        }
    }
}
}  // namespace

int main()
{
    testSegmentedBatchPhases();
    testEncodedNicSobAddress();
    testLongSoChunks(0);
    testLongSoChunks((uint64_t {1} << 15) - 1);
    testLongSoChunks(uint64_t {1} << 15);
    testLongSoChunks((uint64_t {1} << 30) - 1);
    testLongSoChunks(uint64_t {1} << 30);
    testLongSoChunks((uint64_t {1} << 45) - 1);
    testLongSoChunks(uint64_t {1} << 45);
    testLongSoChunks((uint64_t {1} << 60) - 1);
    testWqeBits();
    testTemplateValidation();
    testCapturedCompletionAddress();
    std::cout << "native HCL relocation tests passed\n";
    return 0;
}
