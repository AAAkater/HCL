#include "scal_stream.h"

#include <bits/exception.h>         // for exception
#include <stdexcept>                // for runtime_error
#include "cyclic_buffer_manager.h"  // for CyclicBufferManager
#include "hcl_utils.h"              // for LOG_HCL_TRACE, UNUSED, VERIFY
#include "hcl_log_manager.h"        // for LOG*
#include "scal_wrapper.h"           // for Gen2ArchScalWrapper

#include <algorithm>
#include <cstring>

using namespace hcl;

void* ScalStreamBase::getNextPtr(size_t size)
{
    VERIFY(size % sizeof(uint32_t) == 0);
    m_buffer.resize(size / sizeof(uint32_t));
    return m_buffer.data();
}

ScalStream::ScalStream(ScalJsonNames&       scalNames,
                       const std::string&   schedNameAndStreamNum,
                       const std::string&   schedAndStreamName,
                       Gen2ArchScalWrapper& scalWrapper,
                       CompletionGroup&     cg,
                       unsigned             schedIdx,
                       unsigned             internalStreamIdx,
                       unsigned             archStreamIndex,
                       HclCommandsGen2Arch& commands,
                       CyclicBufferType     type)
: m_scalWrapper(scalWrapper),
  m_schedNameAndStreamNum(schedNameAndStreamNum),
  m_schedAndStreamName(schedAndStreamName),
  m_schedIdx(schedIdx),
  m_internalStreamIdx(internalStreamIdx),
  m_archStreamIndex(archStreamIndex)
{
    m_scalWrapper.initStream(schedNameAndStreamNum,
                             m_streamHandle,
                             m_streamInfo,
                             m_hostCyclicBufferSize,
                             m_bufferHandle,
                             m_bufferInfo);

    LOG_HCL_TRACE(HCL_SCAL,
                  "Created new Stream {} with handle 0x{:x}, and buffer handle 0x{:x}, on host address: 0x{:x}",
                  m_schedNameAndStreamNum,
                  (uint64_t)m_streamHandle,
                  (uint64_t)m_bufferHandle,
                  (uint64_t)m_bufferInfo.host_address);

    m_cyclicBuffer = CyclicBufferFactory::createCyclicBuffer(type,
                                                             this,
                                                             scalNames,
                                                             cg,
                                                             (uint64_t)m_bufferInfo.host_address,
                                                             m_streamInfo,
                                                             m_hostCyclicBufferSize,
                                                             m_schedNameAndStreamNum,
                                                             m_streamHandle,
                                                             m_scalWrapper,
                                                             m_schedIdx,
                                                             commands);
}

void ScalStream::setTargetValue(uint64_t targetValue)
{
    m_cyclicBuffer->setTargetValue(targetValue);
    if (m_nativeCaptureActive)
    {
        m_nativeCaptureTarget = targetValue;
    }
}

void* ScalStream::getNextPtr(size_t size)
{
    void* pointer = m_cyclicBuffer->getNextPtr(size);
    if (m_nativeCaptureActive && !m_cyclicBuffer->isWritingAlignmentPacket())
    {
        m_pendingNativeCommands.push_back(
            {.pointer = static_cast<uint8_t*>(pointer), .size = size, .submitAfter = false});
    }
    return pointer;
}

bool ScalStream::requiresSubmission()
{
    return m_cyclicBuffer->requiresSubmission();
}

void ScalStream::submit()
{
    if (m_nativeCaptureActive && !m_pendingNativeCommands.empty())
    {
        m_pendingNativeCommands.back().submitAfter = true;
    }
    m_cyclicBuffer->submit();
}

void ScalStream::beginNativeCapture()
{
    VERIFY(!m_nativeCaptureActive, "Native HCL capture is already active for {}", m_schedNameAndStreamNum);
    m_nativeCaptureActive  = true;
    m_nativeCaptureStartPi = m_cyclicBuffer->getUnwrappedPi();
    m_nativeCaptureTarget  = 0;
    m_pendingNativeCommands.clear();
    m_pendingNativeRelocations.clear();
}

NativeHclStreamTemplate ScalStream::endNativeCapture()
{
    VERIFY(m_nativeCaptureActive, "Native HCL capture is not active for {}", m_schedNameAndStreamNum);
    const uint64_t bytesAdvanced = m_cyclicBuffer->getUnwrappedPi() - m_nativeCaptureStartPi;
    VERIFY(bytesAdvanced <= m_hostCyclicBufferSize,
           "Native HCL capture for {} advanced {} bytes and overwrote its {} byte CCB",
           m_schedNameAndStreamNum,
           bytesAdvanced,
           m_hostCyclicBufferSize);

    NativeHclStreamTemplate result;
    result.captureTargetValue = m_nativeCaptureTarget;
    result.relocations        = std::move(m_pendingNativeRelocations);
    std::sort(result.relocations.begin(), result.relocations.end(), nativeHclRelocationLess);
    result.commands.reserve(m_pendingNativeCommands.size());
    for (const PendingNativeCommand& pending : m_pendingNativeCommands)
    {
        NativeHclCommand command;
        command.bytes.assign(pending.pointer, pending.pointer + pending.size);
        command.submitAfter = pending.submitAfter;
        result.commands.push_back(std::move(command));
    }

    m_nativeCaptureActive = false;
    m_pendingNativeCommands.clear();
    m_pendingNativeRelocations.clear();
    return result;
}

void ScalStream::cancelNativeCapture()
{
    m_nativeCaptureActive = false;
    m_pendingNativeCommands.clear();
    m_pendingNativeRelocations.clear();
}

void ScalStream::addNativeRelocation(const void*             fieldAddress,
                                     uint8_t                 width,
                                     uint64_t                mask,
                                     uint8_t                 destinationShift,
                                     uint8_t                 sourceShift,
                                     NativeHclRelocationKind kind,
                                     uint32_t capturedAddress)
{
    if (!m_nativeCaptureActive)
    {
        return;
    }
    VERIFY(width == 1 || width == 2 || width == 4 || width == 8, "Unsupported native relocation width {}", width);

    const uintptr_t field = reinterpret_cast<uintptr_t>(fieldAddress);
    for (size_t index = m_pendingNativeCommands.size(); index > 0; --index)
    {
        const PendingNativeCommand& command = m_pendingNativeCommands[index - 1];
        const uintptr_t begin = reinterpret_cast<uintptr_t>(command.pointer);
        const uintptr_t end   = begin + command.size;
        if (field >= begin && field + width <= end)
        {
            m_pendingNativeRelocations.push_back({.commandIndex = static_cast<uint32_t>(index - 1),
                                                  .byteOffset = static_cast<uint32_t>(field - begin),
                                                  .width = width,
                                                  .mask = mask,
                                                  .destinationShift = destinationShift,
                                                  .sourceShift = sourceShift,
                                                  .kind = kind,
                                                  .capturedAddress = capturedAddress});
            return;
        }
    }
    VERIFY(false, "Native relocation field is not inside the most recent captured HCL command");
}

void ScalStream::addNativeRelocationToLastCommandTail(uint32_t                bytesFromEnd,
                                                      uint8_t                 width,
                                                      uint64_t                mask,
                                                      uint8_t                 destinationShift,
                                                      uint8_t                 sourceShift,
                                                      NativeHclRelocationKind kind,
                                                      uint32_t capturedAddress)
{
    if (!m_nativeCaptureActive)
    {
        return;
    }
    VERIFY(!m_pendingNativeCommands.empty(), "Native relocation requires a captured HCL command");
    const PendingNativeCommand& command = m_pendingNativeCommands.back();
    VERIFY(bytesFromEnd >= width && bytesFromEnd <= command.size,
           "Native relocation tail offset {} width {} exceeds command size {}",
           bytesFromEnd,
           width,
           command.size);
    addNativeRelocation(command.pointer + command.size - bytesFromEnd,
                        width,
                        mask,
                        destinationShift,
                        sourceShift,
                        kind,
                        capturedAddress);
}

void ScalStream::replayNativeTemplate(const NativeHclStreamTemplate& commandTemplate,
                                      const NativeHclReplayState&    replayState,
                                      bool                           preserveSubmissions,
                                      bool                           submitAtEnd)
{
    VERIFY(!m_nativeCaptureActive, "Cannot replay while native HCL capture is active");
    VERIFY(validateNativeHclStreamTemplate(commandTemplate),
           "Cannot replay an invalid native HCL stream template");

    setTargetValue(replayState.get(NativeHclRelocationKind::EXTERNAL_COMPLETION_TARGET));
    size_t relocationIndex = 0;
    for (size_t index = 0; index < commandTemplate.commands.size(); ++index)
    {
        const NativeHclCommand& command = commandTemplate.commands[index];
        uint8_t* destination = static_cast<uint8_t*>(m_cyclicBuffer->getNextPtr(command.bytes.size()));
        std::memcpy(destination, command.bytes.data(), command.bytes.size());
        while (relocationIndex < commandTemplate.relocations.size() &&
               commandTemplate.relocations[relocationIndex].commandIndex == index)
        {
            const NativeHclRelocation& relocation = commandTemplate.relocations[relocationIndex++];
            VERIFY(patchNativeHclRelocation(destination, command.bytes.size(), relocation, replayState),
                   "Native HCL relocation cannot be applied");
        }
        if (preserveSubmissions && commandTemplate.commands[index].submitAfter)
        {
            m_cyclicBuffer->submit();
        }
    }
    VERIFY(relocationIndex == commandTemplate.relocations.size(),
           "Native HCL relocation command index is out of range or unsorted");
    if (submitAtEnd)
    {
        m_cyclicBuffer->submit();
    }
}

void ScalStream::replayNativeProgram(const NativeHclProgram& program,
                                     const NativeHclReplayState* states, size_t stateCount)
{
    VERIFY(!m_nativeCaptureActive && states != nullptr && stateCount >= program.stateCount(),
           "Native HCL program has invalid replay state");
    VERIFY(program.alignment() == m_streamInfo.command_alignment && !program.pages().empty(),
           "Native HCL program scheduler configuration changed");
    for (const auto& page : program.pages())
    {
        // A page may contain multiple collectives. It stays live until the last
        // collective using it completes, preserving the CCB's normal recycling.
        setTargetValue(states[page.lastStateIndex].get(NativeHclRelocationKind::EXTERNAL_COMPLETION_TARGET));
        auto* destination = static_cast<uint8_t*>(m_cyclicBuffer->getNextPtr(page.bytes.size()));
        NativeHclProgram::writePage(destination, page, states);
        // Captured publication can protect NIC context/data visibility across
        // microstreams. Do not erase it without an equivalent device edge.
        if (page.publishAfter) m_cyclicBuffer->submit();
    }
}

bool ScalStream::isACcbHalfFullForDeviceBenchMark()
{
    return CyclicBufferManager::isACcbHalfFullForDeviceBenchMark();
}

void ScalStream::disableCcb(bool disable)
{
    m_cyclicBuffer->disableCcb(disable);
}

void ScalStream::dfaLog(hl_logger::LoggerSPtr synDevFailLog)
{
    m_cyclicBuffer->dfaLog(synDevFailLog);
}

ScalStream::~ScalStream()
{
    try
    {
        m_scalWrapper.freeBuffer(m_bufferHandle);
    }
    catch (const std::runtime_error& re)
    {
        LOG_WARN(HCL_SCAL, "Runtime error: {}", re.what());
    }
    catch (const std::exception& ex)
    {
        LOG_WARN(HCL_SCAL, "Error occurred: {}", ex.what());
    }
    catch (...)
    {
        LOG_WARN(HCL_SCAL, "Unknown failure occurred. Possible memory corruption");
    }
}
