#pragma once

#include <cstddef>                   // for size_t
#include <cstdint>                   // for uint32_t, uint64_t
#include <memory>                    // for unique_ptr
#include <string>                    // for string
#include <vector>                    // for vector
#include "scal.h"                    // for scal_buffer_handle_t, scal_buffer_info_t, scal_s...
#include "hl_logger/hllog_core.hpp"  // for hl_logger::LoggerSPtr
#include "infra/scal/gen2_arch_common/cyclic_buffer_factory.h"  // for CyclicBufferFactory
#include "infra/scal/gen2_arch_common/native_hcl_relocation.h"
#include "infra/scal/gen2_arch_common/native_hcl_program.h"
#include "factory_types.h"                                      // for CyclicBufferType

namespace hcl
{
class CompletionGroup;
class CyclicBufferManager;
class Gen2ArchScalWrapper;
class ScalJsonNames;
}  // namespace hcl

class HclCommandsGen2Arch;

namespace hcl
{
/**
 * @brief
 *
 * ScalStreamBase is responsible for encapsulating a buffer. It's used by hcl_packets.h mainly because
 * some clients need the actual buffer (ContextManager, NicPassthroughHandler, unit tests).
 */
class ScalStreamBase
{
public:
    ScalStreamBase()          = default;
    virtual ~ScalStreamBase() = default;

    virtual void*        getNextPtr(size_t size);
    virtual std::string* getStreamName() { return &m_defaultStream; };
    virtual bool         isNativeCaptureActive() const { return false; }
    virtual void addNativeRelocation(const void*,
                                     uint8_t,
                                     uint64_t,
                                     uint8_t,
                                     uint8_t,
                                     NativeHclRelocationKind,
                                     uint32_t = 0)
    {
    }

    std::vector<uint32_t> m_buffer;
    std::string           m_defaultStream = "unknown_stream";
};
/**
 * @brief
 *
 *  ScalStream responsible for managing a cyclic buffer for a given stream name.
 */
class ScalStream : public ScalStreamBase
{
public:
    ScalStream(ScalJsonNames&       scalNames,
               const std::string&   schedNameAndStreamNum,
               const std::string&   schedAndStreamName,
               Gen2ArchScalWrapper& scalWrapper,
               CompletionGroup&     cg,
               unsigned             schedIdx,
               unsigned             internalStreamIdx,
               unsigned             archStreamIndex,
               HclCommandsGen2Arch& commands,
               CyclicBufferType     type);
    ScalStream(ScalStream&&)                 = delete;
    ScalStream(const ScalStream&)            = delete;
    ScalStream& operator=(ScalStream&&)      = delete;
    ScalStream& operator=(const ScalStream&) = delete;
    virtual ~ScalStream();

    void setTargetValue(uint64_t targetValue);

    virtual void* getNextPtr(size_t size) override;

    bool        requiresSubmission();
    void        submit();
    static bool isACcbHalfFullForDeviceBenchMark();
    void        disableCcb(bool disable);
    void        dfaLog(hl_logger::LoggerSPtr synDevFailLog);

    void beginNativeCapture();
    NativeHclStreamTemplate endNativeCapture();
    void cancelNativeCapture();
    void addNativeRelocation(const void*             fieldAddress,
                             uint8_t                 width,
                             uint64_t                mask,
                             uint8_t                 destinationShift,
                             uint8_t                 sourceShift,
                             NativeHclRelocationKind kind,
                             uint32_t capturedAddress = 0) override;
    void addNativeRelocationToLastCommandTail(uint32_t                bytesFromEnd,
                                              uint8_t                 width,
                                              uint64_t                mask,
                                              uint8_t                 destinationShift,
                                              uint8_t                 sourceShift,
                                              NativeHclRelocationKind kind,
                                              uint32_t capturedAddress = 0);
    void replayNativeTemplate(const NativeHclStreamTemplate& commandTemplate,
                              const NativeHclReplayState&    replayState,
                              bool                           preserveSubmissions,
                              bool                           submitAtEnd);
    void replayNativeProgram(const NativeHclProgram& program,
                              const NativeHclReplayState* states, size_t stateCount);
    uint32_t nativeCommandAlignment() const { return m_streamInfo.command_alignment; }
    bool isNativeCaptureActive() const override { return m_nativeCaptureActive; }
    uint64_t getUnwrappedPi() const { return m_cyclicBuffer->getUnwrappedPi(); }
    uint64_t getSubmissionCount() const { return m_cyclicBuffer->getSubmissionCount(); }

    inline unsigned        getUarchStreamIndex() { return m_internalStreamIdx; };
    inline unsigned        getSchedIdx() { return m_schedIdx; };
    inline unsigned        getArchStreamIndex() { return m_archStreamIndex; }
    static inline unsigned getCcbSize() { return m_hostCyclicBufferSize; }
    virtual std::string*   getStreamName() override { return &m_schedNameAndStreamNum; }
    virtual std::string*   getSchedAndStreamName() { return &m_schedAndStreamName; }

private:
    static const uint64_t m_core_counter_max_value = 1 << 16;
    static const uint64_t piQuant                  = 4;

protected:
    scal_stream_handle_t m_streamHandle;
    scal_stream_info_t   m_streamInfo;
    scal_buffer_handle_t m_bufferHandle;
    scal_buffer_info_t   m_bufferInfo;
    Gen2ArchScalWrapper& m_scalWrapper;
    std::string          m_schedNameAndStreamNum;  // For Debug
    std::string          m_schedAndStreamName;     // For Debug
    unsigned             m_schedIdx;
    unsigned             m_internalStreamIdx;  // this is equal to streamIdx in m_streams[schedIdx][streamIdx]
    unsigned             m_archStreamIndex;

    static const uint64_t m_hostCyclicBufferSize = m_core_counter_max_value * piQuant;

    std::unique_ptr<CyclicBufferManager> m_cyclicBuffer;

    struct PendingNativeCommand
    {
        uint8_t* pointer = nullptr;
        size_t   size    = 0;
        bool     submitAfter = false;
    };
    bool                              m_nativeCaptureActive = false;
    uint64_t                          m_nativeCaptureStartPi = 0;
    uint64_t                          m_nativeCaptureTarget = 0;
    std::vector<PendingNativeCommand> m_pendingNativeCommands;
    std::vector<NativeHclRelocation>  m_pendingNativeRelocations;
};
}  // namespace hcl
