#pragma once

#include "native_hcl_relocation.h"
#include <stdexcept>
#include <utility>

namespace hcl
{
struct NativeHclProgramPatch
{
    uint32_t stateIndex = 0;
    NativeHclRelocation field;
};

struct NativeHclProgramPage
{
    std::vector<uint8_t> bytes;
    std::vector<NativeHclProgramPatch> patches;
    uint32_t lastStateIndex = 0;
    bool publishAfter = false;
};

// An immutable stream program can combine any number of collective templates.
// Packet boundaries and relocation validity are resolved once at instantiation.
class NativeHclProgram
{
public:
    using Input = std::pair<const NativeHclStreamTemplate*, uint32_t>;

    static NativeHclProgram prepare(const std::vector<Input>& inputs, size_t alignment,
                                    bool preserveCapturedPublications = true)
    {
        if (inputs.empty() || alignment == 0 || (alignment & (alignment - 1)) != 0)
            throw std::invalid_argument("Invalid native HCL program alignment or inputs");
        NativeHclProgram result;
        result.m_alignment = alignment;
        for (const auto& input : inputs)
        {
            if (input.first == nullptr || !validateNativeHclStreamTemplate(*input.first))
                throw std::invalid_argument("Invalid native HCL program template");
            const auto& commandTemplate = *input.first;
            size_t patch = 0;
            for (size_t commandIndex = 0; commandIndex < commandTemplate.commands.size(); ++commandIndex)
            {
                const auto& command = commandTemplate.commands[commandIndex];
                if (command.bytes.empty() || command.bytes.size() > alignment)
                    throw std::invalid_argument("Native HCL packet exceeds its scheduler alignment");
                if (result.m_pages.empty() || result.m_pages.back().publishAfter ||
                    result.m_pages.back().bytes.size() + command.bytes.size() > alignment)
                    result.m_pages.emplace_back();
                auto& page = result.m_pages.back();
                const size_t offset = page.bytes.size();
                page.bytes.insert(page.bytes.end(), command.bytes.begin(), command.bytes.end());
                page.lastStateIndex = input.second;
                page.publishAfter = preserveCapturedPublications && command.submitAfter;
                while (patch < commandTemplate.relocations.size() &&
                       commandTemplate.relocations[patch].commandIndex == commandIndex)
                {
                    auto field = commandTemplate.relocations[patch++];
                    if (offset > std::numeric_limits<uint32_t>::max() - field.byteOffset)
                        throw std::overflow_error("Native HCL program relocation offset overflow");
                    field.byteOffset += static_cast<uint32_t>(offset);
                    field.commandIndex = 0;
                    page.patches.push_back({input.second, field});
                }
                result.m_bytes += command.bytes.size();
            }
            result.m_stateCount = std::max(result.m_stateCount, size_t(input.second) + 1);
        }
        return result;
    }

    const std::vector<NativeHclProgramPage>& pages() const { return m_pages; }
    size_t stateCount() const { return m_stateCount; }
    size_t alignment() const { return m_alignment; }
    size_t bytes() const { return m_bytes; }

    // Caller checks stateCount once for the program. Each field was validated
    // during prepare, so this hot loop performs only the required epoch update.
    static void writePage(uint8_t* destination,
                          const NativeHclProgramPage& page,
                          const NativeHclReplayState* states)
    {
        std::memcpy(destination, page.bytes.data(), page.bytes.size());
        for (const auto& patch : page.patches)
        {
            const auto& field = patch.field;
            uint64_t word = 0;
            std::memcpy(&word, destination + field.byteOffset, field.width);
            const uint64_t value = states[patch.stateIndex].get(field.kind) >> field.sourceShift;
            word = (word & ~field.mask) | ((value << field.destinationShift) & field.mask);
            std::memcpy(destination + field.byteOffset, &word, field.width);
        }
    }

private:
    std::vector<NativeHclProgramPage> m_pages;
    size_t m_alignment = 0;
    size_t m_stateCount = 0;
    size_t m_bytes = 0;
};
} // namespace hcl
