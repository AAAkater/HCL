/******************************************************************************
 * Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************/

#include "hccl.h"  // for HCCL_VERSION_CODE

#include <common/shim_typedefs.h>  // for PFN_ShimFinish, PFN...
#include <dlfcn.h>                 // for dlclose, dlsym, dle...
#include <climits>                 // for INT_MAX
#include <algorithm>
#include <cstdlib>                 // for getenv
#include <cstring>                 // for strcmp
#include <memory>                  // for shared_ptr
#include <string>                  // for string
#include <mutex>                   // for mutx, unique_lock
#include <atomic>
#include <condition_variable>
#include <cstdio>

#include "common/shim_types.h"       // for SHIM_API_HCCL, SHIM...
#include "dfa_defines.hpp"           // for DfaErrorCode, DfaEr...
#include "hccl_api_funcs.h"          // for hccl_functions_poin...
#include "hccl_communicator.h"       // for hccl_communicator
#include "hccl_context.h"            // for hccl_context, g_hcc...
#include "hccl_helpers.h"            // for to_string, to_hccl_...
#include "hccl_internal_defs.h"      // for hcclOpParams, eHCCL...
#include "hccl_types.h"              // for hcclResult_t, hcclC...
#include "hcl_public_streams.h"      // for tdrDetectionFlag
#include "hcl_types.h"               // for HclConfigType, LOOP...
#include "hcl_utils.h"               // for HCL_API_LOG_ENTRY
#include "hccl_api_inc.h"            // for HCCL_* macros
#include "internal/hccl_internal.h"  // for hcclDFA, hcclDestro...
#include "internal/hccl_tp2_native_graph.h"
#include "hcl_log_manager.h"         // for LOG_ERR, LOG_DEBUG
#include "hccl_gen2_impl.h"          // for Gen2 hccl impl under HclGen2
#include "hccl/tp2_direct_api_context.h"
#include "hccl_wrapper.h"            // for hccl*_Wrapper
#include "fault_tolerance_inc.h"     // for HLFT.* macros
#include "collective_interface/collectives/all_reduce.h"
#include "platform/gen2_arch_common/hccl_device.h"

#include <mutex>

struct HCL_Request;

struct InternalHcclTp2NativeGraph
{
    enum class ReplayPath
    {
        None,
        Prepared,
        Native
    };

    InternalHcclTp2NativeGraph(hccl_communicator*                  communicatorIn,
                               HclCollectiveRoutinesGen2Arch*      routinesIn,
                               HclDynamicCommunicator&             dynamicComm,
                               const void*                         sendbuff,
                               void*                               recvbuff,
                               size_t                              count,
                               hcclDataType_t                      datatype,
                               hcclRedOp_t                         reduceOp,
                               void*                               streamHandleIn,
                               hcclTp2NativeMode                   modeIn,
                               uint8_t                             apiId)
    : communicator(communicatorIn),
      routines(routinesIn),
      params(eHCLAllReduce,
             streamHandleIn,
             reinterpret_cast<uint64_t>(sendbuff),
             reinterpret_cast<uint64_t>(recvbuff),
             count,
             datatype,
             dynamicComm,
             apiId,
             eHCCLAPICall,
             reduceOp),
      streamHandle(streamHandleIn),
      mode(modeIn)
    {
    }

    hccl_communicator*                              communicator = nullptr;
    HclCollectiveRoutinesGen2Arch*                  routines     = nullptr;
    HclCollectiveParams                             params;
    std::unique_ptr<HcclGraph>                      graph;
    HclDeviceControllerGen2Arch::NativeGraphCapture capture;
    HclDeviceControllerGen2Arch::NativeGraphCapture wrapCapture;
    void*                                           streamHandle = nullptr;
    hcclTp2NativeMode                               mode = hcclTp2NativeReduction;
    int                                             tp4Phase = -1;
    hcclTp2NativeGraphState                         state = hcclTp2NativeGraphCreated;
    uint64_t                                        preparedReplayCount = 0;
    uint64_t                                        nativeReplayCount = 0;
    uint64_t                                        nativeWrapReplayCount = 0;
    bool                                            nativeRelocationsValidated = false;
    bool                                            batchOwned = false;
    uint64_t                                        completionRingSize = 0;
    HclDeviceControllerGen2Arch::NativeReplayCounters captureCounters;
    HclDeviceControllerGen2Arch::NativeGraphCapture   diagnosticPreparedCapture;
    bool                                              diagnosticPreparedCaptureValid = false;
    uint64_t                                          diagnosticPreparedExecution = 0;
    ReplayPath                                      replayPath = ReplayPath::None;
    std::mutex                                      mutex;
};

struct InternalHcclTp2NativeBatch
{
    struct Phase
    {
        std::vector<const HclDeviceControllerGen2Arch::NativeGraphCapture*> captures;
        HclDeviceControllerGen2Arch::NativeBatchProgram program;
    };
    using PhaseSet = std::vector<Phase>;
    std::vector<hcclTp2NativeGraphHandle> graphs;
    PhaseSet phases;
    PhaseSet prefixPhases;
    PhaseSet suffixPhases;
    std::vector<hcl::NativeHclReplayState> states;
    std::vector<hcl::syncInfo> dependencies, completions;
    size_t split = 0;
    bool segmentedActive = false;
    bool valid = true;
    std::mutex mutex;
};

static InternalHcclTp2NativeBatch::PhaseSet prepareNativeBatchPhases(
    InternalHcclTp2NativeBatch& batch, size_t offset, size_t count)
{
    VERIFY(count != 0 && offset <= batch.graphs.size() && count <= batch.graphs.size() - offset,
           "Invalid native batch phase range");
    InternalHcclTp2NativeBatch::PhaseSet phases(64);
    for (size_t phase = 0; phase < phases.size(); ++phase)
    {
        auto& prepared = phases[phase];
        prepared.captures.reserve(count);
        for (size_t node = 0; node < count; ++node)
        {
            auto graph = batch.graphs[offset + node];
            prepared.captures.push_back(hcl::nativeHclBatchUsesWrapCapture(phase, node)
                                           ? &graph->wrapCapture : &graph->capture);
        }
        prepared.program = batch.graphs.front()->routines->prepareNativeBatch(prepared.captures);
    }
    return phases;
}

static void dumpNativeCaptureDiff(
    const HclDeviceControllerGen2Arch::NativeGraphCapture& captured,
    const HclDeviceControllerGen2Arch::NativeGraphCapture& prepared,
    uint64_t                                                fromExecution,
    uint64_t                                                toExecution)
{
    std::fprintf(stderr,
                 "NATIVE_HCL_DIFF summary executions=%llu->%llu streams=%zu/%zu commands=%zu/%zu "
                 "targets=%llu,%llu/%llu,%llu "
                 "completion_addresses=0x%x,0x%x/0x%x,0x%x\n",
                 static_cast<unsigned long long>(fromExecution),
                 static_cast<unsigned long long>(toExecution),
                 captured.streams.size(),
                 prepared.streams.size(),
                 captured.commandCount(),
                 prepared.commandCount(),
                 static_cast<unsigned long long>(captured.externalTargetDelta),
                 static_cast<unsigned long long>(captured.internalTargetDelta),
                 static_cast<unsigned long long>(prepared.externalTargetDelta),
                 static_cast<unsigned long long>(prepared.internalTargetDelta),
                 captured.capturedExternalCompletionAddress,
                 captured.capturedInternalCompletionAddress,
                 prepared.capturedExternalCompletionAddress,
                 prepared.capturedInternalCompletionAddress);
    size_t differences = 0;
    constexpr size_t maxPrintedDifferences = 512;
    for (const auto& oldStream : captured.streams)
    {
        const auto newStream = std::find_if(
            prepared.streams.begin(), prepared.streams.end(), [&](const auto& candidate) {
                return candidate.streamIndex == oldStream.streamIndex;
            });
        if (newStream == prepared.streams.end())
        {
            std::fprintf(stderr,
                         "NATIVE_HCL_DIFF missing_stream stream=%u\n",
                         static_cast<unsigned>(oldStream.streamIndex));
            continue;
        }
        std::fprintf(stderr,
                     "NATIVE_HCL_DIFF stream=%u commands=%zu/%zu relocations=%zu/%zu\n",
                     static_cast<unsigned>(oldStream.streamIndex),
                     oldStream.commandTemplate.commands.size(),
                     newStream->commandTemplate.commands.size(),
                     oldStream.commandTemplate.relocations.size(),
                     newStream->commandTemplate.relocations.size());
        const size_t commands = std::min(oldStream.commandTemplate.commands.size(),
                                         newStream->commandTemplate.commands.size());
        for (size_t commandIndex = 0; commandIndex < commands; ++commandIndex)
        {
            const auto& oldBytes = oldStream.commandTemplate.commands[commandIndex].bytes;
            const auto& newBytes = newStream->commandTemplate.commands[commandIndex].bytes;
            if (oldBytes.size() != newBytes.size())
            {
                std::fprintf(stderr,
                             "NATIVE_HCL_DIFF size stream=%u command=%zu bytes=%zu/%zu\n",
                             static_cast<unsigned>(oldStream.streamIndex),
                             commandIndex,
                             oldBytes.size(),
                             newBytes.size());
            }
            const size_t bytes = std::min(oldBytes.size(), newBytes.size());
            for (size_t offset = 0; offset < bytes; offset += sizeof(uint32_t))
            {
                const size_t width = std::min(sizeof(uint32_t), bytes - offset);
                uint32_t oldWord = 0;
                uint32_t newWord = 0;
                std::memcpy(&oldWord, oldBytes.data() + offset, width);
                std::memcpy(&newWord, newBytes.data() + offset, width);
                if (oldWord == newWord)
                {
                    continue;
                }
                if (differences < maxPrintedDifferences)
                {
                    std::fprintf(stderr,
                                 "NATIVE_HCL_DIFF word stream=%u command=%zu offset=%zu old=0x%08x "
                                 "new=0x%08x xor=0x%08x\n",
                                 static_cast<unsigned>(oldStream.streamIndex),
                                 commandIndex,
                                 offset,
                                 oldWord,
                                 newWord,
                                 oldWord ^ newWord);
                    for (const auto& relocation : oldStream.commandTemplate.relocations)
                    {
                        const size_t relocationBegin = relocation.byteOffset;
                        const size_t relocationEnd = relocation.byteOffset + relocation.width;
                        if (relocation.commandIndex == commandIndex && relocationBegin < offset + width &&
                            relocationEnd > offset)
                        {
                            std::fprintf(stderr,
                                         "NATIVE_HCL_DIFF relocation stream=%u command=%zu offset=%u width=%u "
                                         "mask=0x%llx kind=%u\n",
                                         static_cast<unsigned>(oldStream.streamIndex),
                                         commandIndex,
                                         relocation.byteOffset,
                                         static_cast<unsigned>(relocation.width),
                                         static_cast<unsigned long long>(relocation.mask),
                                         static_cast<unsigned>(relocation.kind));
                        }
                    }
                }
                ++differences;
            }
        }
    }
    std::fprintf(stderr, "NATIVE_HCL_DIFF total_changed_words=%zu\n", differences);
    std::fflush(stderr);
}

// ------------------------------------------------------------------------------------------------
// Global context singleton objects.
//

#define HCCL_API_CALL __attribute__((visibility("default")))

hccl_context hccl_ctx;

// DFA vars
DfaPhase   g_dfaPhase = DfaPhase::NONE;
std::mutex g_dfaMutex;

// Fault tolerance vars
std::atomic<bool> g_faultsCheckStopApi {false};  // To check conditions for faults API stop handling only when necessary
std::atomic<uint32_t>   g_faultsStopAllApi {false};  // To track stop API requests
std::condition_variable g_faultsStopAllApiCv;        // CV to block user API threads
std::mutex              g_faultsStopAllApiMutex;     // Mutex for condition variable

// Called to check if to stop all API's during fault tolerance handling from hccl.cpp and hccl_wrapper.cpp
void checkFaultToleranceStopApi()
{
    LOG_DEBUG(HCL_FAILOVER, "{}: Stop API check", __func__);
    LOG_DEBUG(HCL_API, "{}: Stop API check", __func__);
    std::unique_lock<std::mutex> lk(g_faultsStopAllApiMutex);
    LOG_DEBUG(HCL_FAILOVER, "{}: Before CV wait, g_faultsStopAllApi={}", __func__, g_faultsStopAllApi.load());
    LOG_DEBUG(HCL_API, "{}: Before CV wait, g_faultsStopAllApi={}", __func__, g_faultsStopAllApi.load());
    g_faultsStopAllApiCv.wait(lk,
                              [] { return (g_faultsStopAllApi.load() == 0); }); /* Block if g_faultsStopAllApi != 0 */
    LOG_INFO(HCL_FAILOVER,
             "{}: After CV wait, User API thread is unblocked, g_faultsCheckStopApi={}",
             __func__,
             g_faultsCheckStopApi.load());
    LOG_INFO(HCL_API,
             "{}: After CV wait, User API thread is unblocked, g_faultsCheckStopApi={}",
             __func__,
             g_faultsCheckStopApi.load());
}

static hcclResult_t syncHCLStreamHandle(synStreamHandle stream_handle)
{
    const synStatus    status      = synStreamSyncHCLStreamHandle(stream_handle);
    const hcclResult_t hccl_status = to_hccl_result(status);
    if (hccl_status != hcclSuccess) return hcclInvalidUsage;

    return hcclSuccess;
}

hcclResult_t HCCL_API_CALL hcclGetVersion_Original(int* version)
{
    return hcclGetVersion_Wrapper(version);
}

hcclResult_t HCCL_API_CALL hcclGetUniqueId_Original(hcclUniqueId* uniqueId)
{
    return hcclGetUniqueId_Wrapper(uniqueId);
}

hcclResult_t HCCL_API_CALL hcclCommInitRank_Original(hcclComm_t* comm, int nranks, hcclUniqueId& commId, int rank)
{
    return hcclCommInitRank_Wrapper(comm, nranks, commId, rank);
}

hcclResult_t HCCL_API_CALL hcclCommInitAll_Original(hcclComm_t* comm, int ndev, const int* devlist)
{
    return hcclCommInitAll_Wrapper(comm, ndev, devlist);
}

hcclResult_t HCCL_API_CALL hcclCommFinalize_Original(hcclComm_t comm)
{
    return hcclCommFinalize_Wrapper(comm);
}

static bool hcclIsACcbHalfFull_Original(const unsigned archStreamIdx)
{
    HCCL_TRY
    const bool res = hccl_device()->isACcbHalfFullForDeviceBenchMark(archStreamIdx);
    HCL_API_EXIT(res)
}

static void hcclSetTraceMarker_Original(const synStreamHandle stream_handle, uint32_t val)
{
    hccl_device()->setTraceMarker(stream_handle, val);
}

hcclResult_t HCCL_API_CALL hcclCommDestroy_Original(hcclComm_t comm)
{
    return hcclCommDestroy_Wrapper(comm);
}

hcclResult_t HCCL_API_CALL hcclCommAbort_Original(hcclComm_t comm)
{
    return hcclCommAbort_Wrapper(comm);
}

HCCL_API_CALL const char* hcclGetErrorString_Original(hcclResult_t result)
{
    return hcclGetErrorString_Wrapper(result);
}

hcclResult_t HCCL_API_CALL hcclCommGetAsyncError_Original(hcclComm_t comm, hcclResult_t* asyncError)
{
    return hcclCommGetAsyncError_Wrapper(comm, asyncError);
}

hcclResult_t HCCL_API_CALL hcclCommCount_Original(hcclComm_t comm, int* count)
{
    return hcclCommCount_Wrapper(comm, count);
}

hcclResult_t HCCL_API_CALL hcclCommSynDevice_Original(hcclComm_t comm, int* device)
{
    return hcclCommSynDevice_Wrapper(comm, device);
}

hcclResult_t HCCL_API_CALL hcclCommUserRank_Original(hcclComm_t comm, int* rank)
{
    return hcclCommUserRank_Wrapper(comm, rank);
}

int HCCL_API_CALL hcclLookupDMABuff_Original(uint64_t addr, uint64_t size, int* fd)
{
    return hcclLookupDMABuff_Wrapper(addr, size, fd);
}

hcclResult_t HCCL_API_CALL hcclReduceScatter_Original(const void*     sendbuff,
                                                      void*           recvbuff,
                                                      size_t          recvcount,
                                                      hcclDataType_t  datatype,
                                                      hcclRedOp_t     reduceOp,
                                                      hcclComm_t      comm,
                                                      synStreamHandle stream_handle)
{
    return hcclReduceScatter_Wrapper(sendbuff, recvbuff, recvcount, datatype, reduceOp, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclAllReduce_Original(const void*     sendbuff,
                                                  void*           recvbuff,
                                                  size_t          count,
                                                  hcclDataType_t  datatype,
                                                  hcclRedOp_t     reduceOp,
                                                  hcclComm_t      comm,
                                                  synStreamHandle stream_handle)
{
    return hcclAllReduce_Wrapper(sendbuff, recvbuff, count, datatype, reduceOp, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclReduce_Original(const void*     sendbuff,
                                               void*           recvbuff,
                                               size_t          count,
                                               hcclDataType_t  datatype,
                                               hcclRedOp_t     reduceOp,
                                               int             root,
                                               hcclComm_t      comm,
                                               synStreamHandle stream_handle)
{
    return hcclReduce_Wrapper(sendbuff, recvbuff, count, datatype, reduceOp, root, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclBcast_Original(void*           buff,
                                              size_t          count,
                                              hcclDataType_t  datatype,
                                              int             root,
                                              hcclComm_t      comm,
                                              synStreamHandle stream_handle)
{
    return hcclBroadcast(buff, buff, count, datatype, root, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclBroadcast_Original(const void*     sendbuff,
                                                  void*           recvbuff,
                                                  size_t          count,
                                                  hcclDataType_t  datatype,
                                                  int             root,
                                                  hcclComm_t      comm,
                                                  synStreamHandle stream_handle)
{
    return hcclBroadcast_Wrapper(sendbuff, recvbuff, count, datatype, root, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclAllGather_Original(const void*     sendbuff,
                                                  void*           recvbuff,
                                                  size_t          sendcount,
                                                  hcclDataType_t  datatype,
                                                  hcclComm_t      comm,
                                                  synStreamHandle stream_handle)
{
    return hcclAllGather_Wrapper(sendbuff, recvbuff, sendcount, datatype, comm, stream_handle);
}

hcclResult_t hcclAlltoAll_Original(const void*     sendbuff,
                                   void*           recvbuff,
                                   size_t          count,
                                   hcclDataType_t  datatype,
                                   hcclComm_t      comm,
                                   synStreamHandle stream_handle)
{
    return hcclAlltoAll_Wrapper(sendbuff, recvbuff, count, datatype, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclBarrier_Original([[maybe_unused]] hcclComm_t      comm_handle,
                                                [[maybe_unused]] synStreamHandle stream_handle)
{
    HCCL_TRY
    LOG_ERR(HCL_API, "Unsupported API for Gen2");
    HCCL_API_EXIT(hcclUnsupported)
}

hcclResult_t HCCL_API_CALL hcclSend_Original(const void*     sendbuff,
                                             size_t          count,
                                             hcclDataType_t  datatype,
                                             int             peer,
                                             hcclComm_t      comm,
                                             synStreamHandle stream_handle)
{
    return hcclSend_Wrapper(sendbuff, count, datatype, peer, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclRecv_Original(void*           recvbuff,
                                             size_t          count,
                                             hcclDataType_t  datatype,
                                             int             peer,
                                             hcclComm_t      comm,
                                             synStreamHandle stream_handle)
{
    return hcclRecv_Wrapper(recvbuff, count, datatype, peer, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclGroupStart_Original()
{
    return hcclGroupStart_Wrapper();
}

hcclResult_t HCCL_API_CALL hcclGroupEnd_Original()
{
    return hcclGroupEnd_Wrapper();
}

hcclResult_t hcclInitDevice_Original([[maybe_unused]] const synDeviceId deviceId)
{
    HCCL_TRY
    hcclResult_t status = hccl_ctx.init_device(hccl_ctx.generateApiId());
    HCCL_API_EXIT(status)
}

hcclResult_t hcclDestroyDevice_Original(const synDeviceId deviceId)
{
    return hcclDestroyDevice_Wrapper(deviceId);
}

hcclResult_t hcclEventRecord_Original(hcclEventHandle* eventHandle, synStreamHandle streamHandle)
{
    HCCL_TRY
    RETURN_ON_INVALID_STREAM(streamHandle);
    hcclResult_t status = hccl_device()->networkFlush((HCL_Request*)eventHandle, streamHandle);
    HCCL_API_EXIT(status)
}

hcclResult_t hcclSynchronizeEvent_Original([[maybe_unused]] const hcclEventHandle& eventHandle,
                                           [[maybe_unused]] uint64_t               microSeconds)
{
    HCCL_TRY
    LOG_ERR(HCL_API, "Unsupported API for Gen2");
    HCCL_API_EXIT(hcclUnsupported)
}

hcclResult_t hcclFlushSubmissions_Original(synStreamHandle streamHandle)
{
    HCCL_TRY
    RETURN_ON_INVALID_STREAM(streamHandle);
    hccl_device()->waitForAllEvents(synStreamGetPhysicalQueueOffset(streamHandle), false);
    HCCL_API_EXIT(hcclSuccess)
}

hcclResult_t hcclFlushSubmissionsAllStreams_Original()
{
    HCCL_TRY
    hccl_device()->waitForAllEvents(false);
    HCCL_API_EXIT(hcclSuccess)
}

hcclResult_t hcclSynchronizeStream_Original(synStreamHandle streamHandle)
{
    HCCL_TRY
    RETURN_ON_INVALID_STREAM(streamHandle);
    hccl_device()->waitForAllEvents(synStreamGetPhysicalQueueOffset(streamHandle), true);
    HCCL_API_EXIT(hcclSuccess)
}

hcclResult_t hcclSynchronizeAllStreams_Original()
{
    HCCL_TRY
    hccl_device()->waitForAllEvents(true);
    HCCL_API_EXIT(hcclSuccess)
}

hcclResult_t hcclDFA_Original(DfaStatus& dfaStatus, void (*dfaLogFunc)(int, const char*))
{
    if (dfaStatus.hasError(DfaErrorCode::scalTdrFailed))
    {
        tdrDetectionFlag = true;
    }

    LOG_INFO(HCL, "DFA asked to dump info");
    dfaLogFunc(HLLOG_LEVEL_INFO, "HCL DFA not implemented");
    return hcclSuccess;
}

hcclResult_t hcclDfaUpdateState_Original(DfaPhase dfaPhase)
{
    bool updateErr = false;
    switch (dfaPhase)
    {
        case DfaPhase::NONE:
            if (g_dfaPhase == DfaPhase::STARTED)
            {
                g_dfaMutex.unlock();
            }
            else
            {
                updateErr = true;
            }
            setGlobalDfaStatus(hcclSuccess);
            break;

        case DfaPhase::STARTED:
        {
            if (g_dfaPhase != DfaPhase::STARTED)
            {
                g_dfaMutex.lock();
            }
            else
            {
                updateErr = true;
            }
            setGlobalDfaStatus(hcclResult_t::hcclInternalError);
            break;
        }

        case DfaPhase::ENDED:
            if (g_dfaPhase == DfaPhase::STARTED)
            {
                g_dfaMutex.unlock();
            }
            else
            {
                updateErr = true;
            }
            setGlobalDfaStatus(hcclResult_t::hcclInternalError);
            break;
    }

    if (updateErr)
    {
        LOG_ERR(HCL, "Unexpected Dfa phase update. Old {} new {}", g_dfaPhase, dfaPhase);
    }
    LOG_INFO(HCL_API, "Dfa state change. Old {} new {}", g_dfaPhase, dfaPhase);
    g_dfaPhase = dfaPhase;
    return hcclSuccess;
}

hcclResult_t hcclGetVersionString_Original(char* pVersion, const unsigned len)
{
    HCCL_TRY
    RETURN_ON_NULL_ARG(pVersion);

    getHclVersion(pVersion, len);
    LOG_DEBUG(HCL_API, "HCCL Version String is: {}", pVersion);
    HCCL_API_EXIT(hcclSuccess)
}

hcclResult_t HCCL_API_CALL hcclDeviceInit_Original(void* device, void* context)
{
    return hcclDeviceInit_Wrapper(device, context);
}

static struct hccl_functions_pointers default_functions_pointers_table = {
    .pfn_hcclGetVersion                 = hcclGetVersion_Original,
    .pfn_hcclGetUniqueId                = hcclGetUniqueId_Original,
    .pfn_hcclCommInitRank               = hcclCommInitRank_Original,
    .pfn_hcclCommInitAll                = hcclCommInitAll_Original,
    .pfn_hcclCommDestroy                = hcclCommDestroy_Original,
    .pfn_hcclCommAbort                  = hcclCommAbort_Original,
    .pfn_hcclGetErrorString             = hcclGetErrorString_Original,
    .pfn_hcclCommGetAsyncError          = hcclCommGetAsyncError_Original,
    .pfn_hcclCommCount                  = hcclCommCount_Original,
    .pfn_hcclCommSynDevice              = hcclCommSynDevice_Original,
    .pfn_hcclCommUserRank               = hcclCommUserRank_Original,
    .pfn_hcclLookupDMABuff              = hcclLookupDMABuff_Original,
    .pfn_hcclReduce                     = hcclReduce_Original,
    .pfn_hcclBcast                      = hcclBcast_Original,
    .pfn_hcclBroadcast                  = hcclBroadcast_Original,
    .pfn_hcclAllReduce                  = hcclAllReduce_Original,
    .pfn_hcclReduceScatter              = hcclReduceScatter_Original,
    .pfn_hcclAllGather                  = hcclAllGather_Original,
    .pfn_hcclAlltoAll                   = hcclAlltoAll_Original,
    .pfn_hcclBarrier                    = hcclBarrier_Original,
    .pfn_hcclSend                       = hcclSend_Original,
    .pfn_hcclRecv                       = hcclRecv_Original,
    .pfn_hcclGroupStart                 = hcclGroupStart_Original,
    .pfn_hcclGroupEnd                   = hcclGroupEnd_Original,
    .pfn_hcclInitDevice                 = hcclInitDevice_Original,
    .pfn_hcclDestroyDevice              = hcclDestroyDevice_Original,
    .pfn_hcclEventRecord                = hcclEventRecord_Original,
    .pfn_hcclSynchronizeEvent           = hcclSynchronizeEvent_Original,
    .pfn_hcclFlushSubmissions           = hcclFlushSubmissions_Original,
    .pfn_hcclFlushSubmissionsAllStreams = hcclFlushSubmissionsAllStreams_Original,
    .pfn_hcclSynchronizeStream          = hcclSynchronizeStream_Original,
    .pfn_hcclSynchronizeAllStreams      = hcclSynchronizeAllStreams_Original,
    .pfn_hcclDFA                        = hcclDFA_Original,
    .pfn_hcclDfaUpdateState             = hcclDfaUpdateState_Original,
    .pfn_hcclGetVersionString           = hcclGetVersionString_Original,
    .pfn_hcclCommFinalize               = hcclCommFinalize_Original,
    .pfn_hcclDeviceInit                 = hcclDeviceInit_Original};
// functions_pointers_table will maintain the current functions pointers table
// Initialized to the original functions
static struct hccl_functions_pointers* functions_pointers_table = &default_functions_pointers_table;

static void*                s_shimLibHandle       = nullptr;
static PFN_ShimGetFunctions s_pfnShimGetFunctions = nullptr;
static PFN_ShimFinish       s_pfnShimFinish       = nullptr;

static void HCCL_StartShim(void)
{
    if (s_shimLibHandle == nullptr)
    {
        const char* envValue = getenv("HABANA_SHIM_DISABLE");
        if (envValue == nullptr || strcmp(envValue, "1"))
        {
            LOG_DEBUG(HCL, "Engaging shim layer");
            s_shimLibHandle = dlopen(SHIM_LIB_NAME, RTLD_LAZY);
            if (s_shimLibHandle != nullptr)
            {
                void* fn = dlsym(s_shimLibHandle, SHIM_GET_FUNCTIONS);
                if (fn != nullptr)
                {
                    LOG_DEBUG(HCL, "shim layer initialized successfully");
                    s_pfnShimGetFunctions = reinterpret_cast<PFN_ShimGetFunctions>(fn);
                    s_pfnShimFinish       = reinterpret_cast<PFN_ShimFinish>(dlsym(s_shimLibHandle, SHIM_FINISH));
                    /*
                     * start/stop profiling is not supported at the moment.
                     * Currently, we call ShimGetFunctions only once in the initialization
                     * To support profiling during execution,
                     * we will have to call it before every (or specific) API call
                     */
                    functions_pointers_table = static_cast<hccl_functions_pointers*>(
                        s_pfnShimGetFunctions(SHIM_API_HCCL, functions_pointers_table));
                }
                else
                {
                    LOG_ERR(HCL, "shim layer entry point function was not found");
                    dlclose(s_shimLibHandle);
                    s_shimLibHandle = nullptr;
                }
            }
            else
            {
                LOG_ERR(HCL, "Could not load shim layer binary - {}", SHIM_LIB_NAME);
                LOG_ERR(HCL, "dlerror: {} ", dlerror());
            }
        }
    }
}

static void HCCL_StopShim(void)
{
    functions_pointers_table = &default_functions_pointers_table;
    if (s_shimLibHandle != nullptr)
    {
        if (s_pfnShimFinish != nullptr)
        {
            s_pfnShimFinish(SHIM_API_HCCL);
        }
        dlclose(s_shimLibHandle);
        s_shimLibHandle = nullptr;
    }
}

namespace HclGen2
{

namespace
{
enum class Tp2DedicatedMode
{
    none,
    reduction,
    exchange,
};

thread_local Tp2DedicatedMode s_tp2DedicatedMode = Tp2DedicatedMode::none;

class ScopedTp2DedicatedDirectCall
{
public:
    explicit ScopedTp2DedicatedDirectCall(Tp2DedicatedMode mode) : m_previous(s_tp2DedicatedMode)
    {
        s_tp2DedicatedMode = mode;
    }

    ~ScopedTp2DedicatedDirectCall()
    {
        s_tp2DedicatedMode = m_previous;
    }

private:
    Tp2DedicatedMode m_previous;
};
}  // namespace

bool isTp2DedicatedDirectCall()
{
    return s_tp2DedicatedMode != Tp2DedicatedMode::none;
}

bool isTp2DedicatedExchangeCall()
{
    return s_tp2DedicatedMode == Tp2DedicatedMode::exchange;
}

hcclResult_t HCCL_API_CALL hcclGetVersion_impl(int* version)
{
    HCL_API_LOG_ENTRY("(&version={:p})", (void*)version);
    return (*functions_pointers_table->pfn_hcclGetVersion)(version);
}

hcclResult_t HCCL_API_CALL hcclGetUniqueId_impl(hcclUniqueId* uniqueId)
{
    HCL_API_LOG_ENTRY("(&uniqueId={:p})", (void*)uniqueId);
    return (*functions_pointers_table->pfn_hcclGetUniqueId)(uniqueId);
}

hcclResult_t HCCL_API_CALL hcclCommInitRank_impl(hcclComm_t* comm, int nranks, hcclUniqueId commId, int rank)
{
    HCCL_StartShim();
    return (*functions_pointers_table->pfn_hcclCommInitRank)(comm, nranks, commId, rank);
}

hcclResult_t HCCL_API_CALL hcclCommInitAll_impl(hcclComm_t* comm, int ndev, const int* devlist)
{
    HCL_API_LOG_ENTRY("(&comm={:p}, ndev={}, &devlist={})", (void*)comm, ndev, (void*)devlist);
    HCCL_StartShim();

    return (*functions_pointers_table->pfn_hcclCommInitAll)(comm, ndev, devlist);
}

hcclResult_t HCCL_API_CALL hcclCommFinalize_impl(hcclComm_t comm)
{
    HCL_API_LOG_ENTRY("(&comm={:p})", (void*)comm);
    return (*functions_pointers_table->pfn_hcclCommFinalize)(comm);
}

bool HCCL_API_CALL hcclIsACcbHalfFull_impl(const unsigned archStreamIdx)
{
    HCCL_TRY
    const bool res = hcclIsACcbHalfFull_Original(archStreamIdx);
    HCL_API_EXIT(res)
}

void HCCL_API_CALL hcclSetTraceMarker_impl(const synStreamHandle stream_handle, uint32_t val)
{
    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return;
    hcclSetTraceMarker_Original(stream_handle, val);
}

hcclResult_t HCCL_API_CALL hcclCommDestroy_impl(hcclComm_t comm)
{
    HCL_API_LOG_ENTRY("(&comm={:p})", (void*)comm);
    hcclResult_t status = (*functions_pointers_table->pfn_hcclCommDestroy)(comm);
    HCCL_StopShim();

    return status;
}

hcclResult_t HCCL_API_CALL hcclCommAbort_impl(hcclComm_t comm)
{
    HCL_API_LOG_ENTRY("(&comm={:p})", (void*)comm);
    return (*functions_pointers_table->pfn_hcclCommAbort)(comm);
}

HCCL_API_CALL const char* hcclGetErrorString_impl(hcclResult_t result)
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclGetErrorString)(result);
}

HCCL_API_CALL const char* hcclGetLastErrorMessage_impl()
{
    return hcclGetLastErrorMessage_Wrapper();
}

hcclResult_t HCCL_API_CALL hcclCommGetAsyncError_impl(hcclComm_t comm, hcclResult_t* asyncError)
{
    HCL_API_LOG_ENTRY("(&comm={:p})", (void*)comm);
    return (*functions_pointers_table->pfn_hcclCommGetAsyncError)(comm, asyncError);
}

HCCL_API_CALL const char* hcclCommGetAsyncErrorMessage_impl(hcclComm_t comm)
{
    HCL_API_LOG_ENTRY("(&comm={:p})", (void*)comm);
    return hcclCommGetAsyncErrorMessage_Wrapper(comm);
}

hcclResult_t HCCL_API_CALL hcclCommCount_impl(hcclComm_t comm, int* count)
{
    HCL_API_LOG_ENTRY("(&comm={:p}, &count={:p})", (void*)comm, (void*)count);
    return (*functions_pointers_table->pfn_hcclCommCount)(comm, count);
}

hcclResult_t HCCL_API_CALL hcclCommSynDevice_impl(hcclComm_t comm, int* device)
{
    HCL_API_LOG_ENTRY("(&comm={:p}, &device={:p})", (void*)comm, (void*)device);
    return (*functions_pointers_table->pfn_hcclCommSynDevice)(comm, device);
}

hcclResult_t HCCL_API_CALL hcclCommUserRank_impl(hcclComm_t comm, int* rank)
{
    HCL_API_LOG_ENTRY("(&comm={:p}, &rank={:p})", (void*)comm, (void*)rank);
    return (*functions_pointers_table->pfn_hcclCommUserRank)(comm, rank);
}

int HCCL_API_CALL hcclLookupDMABuff_impl(uint64_t addr, uint64_t size, int* fd)
{
    HCL_API_LOG_ENTRY("(&addr={:p}, &size={:p})", (void*)addr, (void*)size);
    return (*functions_pointers_table->pfn_hcclLookupDMABuff)(addr, size, fd);
}

hcclResult_t HCCL_API_CALL hcclReduceScatter_impl(const void*     sendbuff,
                                                  void*           recvbuff,
                                                  size_t          recvcount,
                                                  hcclDataType_t  datatype,
                                                  hcclRedOp_t     reduceOp,
                                                  hcclComm_t      comm,
                                                  synStreamHandle stream_handle)
{
    RETURN_ON_INVALID_DATA_TYPE(datatype);

    auto* hccl_comm = hccl_ctx.communicator(comm);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);
    HCCL_CHECK_STOP_COLL_API_COMM_UNTIL(hccl_comm);

    hccl_comm->incCollectiveCtr();

    HCL_API_LOG_ENTRY(
        "rank={}/{}, oam={}, (sendbuff={:p}, recvbuff={:p}, recvcount={}, datatype={}, reduceOp={}, uniqId={}, "
        "stream_handle={:p}) - collective#=0x{:x}",
        hccl_comm->user_rank(),
        hccl_comm->getCommSize(),
        hccl_device()->getHwModuleId(),
        (void*)sendbuff,
        (void*)recvbuff,
        recvcount,
        to_string(datatype),
        to_string(reduceOp),
        hccl_comm->getCommUniqueId(),
        (void*)stream_handle,
        hccl_comm->getCollectiveCtr());

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    LOG_SYNC_DBG(ReduceScatter,
                 "#Lines: 1 {:#x} {:#x} {:#x} {:#x} {:#x}",
                 TO64(stream_handle),
                 TO64(hccl_comm),
                 TO64(sendbuff),
                 TO64(recvbuff),
                 recvcount * hccl_data_type_elem_size(datatype));

    return (*functions_pointers_table
                 ->pfn_hcclReduceScatter)(sendbuff, recvbuff, recvcount, datatype, reduceOp, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclAllReduce_impl(const void*     sendbuff,
                                              void*           recvbuff,
                                              size_t          count,
                                              hcclDataType_t  datatype,
                                              hcclRedOp_t     reduceOp,
                                              hcclComm_t      comm,
                                              synStreamHandle stream_handle)
{
    RETURN_ON_INVALID_DATA_TYPE(datatype);

    auto* hccl_comm = hccl_ctx.communicator(comm);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);
    HCCL_CHECK_STOP_COLL_API_COMM_UNTIL(hccl_comm);

    hccl_comm->incCollectiveCtr();

    HCL_API_LOG_ENTRY("rank={}/{}, oam={}, (sendbuff={:p}, recvbuff={:p}, count={}, datatype={}, reduceOp={}, "
                      "uniqId={}, stream_handle={:p}) - "
                      "collective#=0x{:x}",
                      hccl_comm->user_rank(),
                      hccl_comm->getCommSize(),
                      hccl_device()->getHwModuleId(),
                      (void*)sendbuff,
                      (void*)recvbuff,
                      count,
                      to_string(datatype),
                      to_string(reduceOp),
                      hccl_comm->getCommUniqueId(),
                      (void*)stream_handle,
                      hccl_comm->getCollectiveCtr());

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    LOG_SYNC_DBG(AllReduce,
                 "#Lines: 1 {:#x} {:#x} {:#x} {:#x} {:#x}",
                 TO64(stream_handle),
                 TO64(hccl_comm),
                 TO64(sendbuff),
                 TO64(recvbuff),
                 count * hccl_data_type_elem_size(datatype));

    return (*functions_pointers_table
                 ->pfn_hcclAllReduce)(sendbuff, recvbuff, count, datatype, reduceOp, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclTp2DirectAllReduce_impl(const void*     sendbuff,
                                                       void*           recvbuff,
                                                       size_t          count,
                                                       hcclDataType_t  datatype,
                                                       hcclRedOp_t     reduceOp,
                                                       hcclComm_t      comm,
                                                       synStreamHandle stream_handle)
{
    if (!GCFG_HCL_TP2_DEDICATED_DIRECT_API.value() || sendbuff == recvbuff || datatype != hcclBfloat16 ||
        reduceOp != hcclSum || count == 0)
    {
        return hcclInvalidArgument;
    }
    const uint64_t maxCount = GCFG_HCL_TP2_DIRECT_MAX_COUNT.value();
    if (maxCount > 0 && count > maxCount)
    {
        return hcclInvalidArgument;
    }

    ScopedTp2DedicatedDirectCall scope(Tp2DedicatedMode::reduction);
    return hcclAllReduce_impl(sendbuff, recvbuff, count, datatype, reduceOp, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclTp2DirectExchange_impl(const void*     sendbuff,
                                                      void*           recvbuff,
                                                      size_t          count,
                                                      hcclDataType_t  datatype,
                                                      hcclRedOp_t     reduceOp,
                                                      hcclComm_t      comm,
                                                      synStreamHandle stream_handle)
{
    if (!GCFG_HCL_TP2_DEDICATED_DIRECT_API.value() || sendbuff == recvbuff || datatype != hcclBfloat16 ||
        reduceOp != hcclSum || count == 0)
    {
        return hcclInvalidArgument;
    }
    const uint64_t maxCount = GCFG_HCL_TP2_DIRECT_MAX_COUNT.value();
    if (maxCount > 0 && count > maxCount)
    {
        return hcclInvalidArgument;
    }

    ScopedTp2DedicatedDirectCall scope(Tp2DedicatedMode::exchange);
    return hcclAllReduce_impl(sendbuff, recvbuff, count, datatype, reduceOp, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclReduce_impl(const void*     sendbuff,
                                           void*           recvbuff,
                                           size_t          count,
                                           hcclDataType_t  datatype,
                                           hcclRedOp_t     reduceOp,
                                           int             root,
                                           hcclComm_t      comm,
                                           synStreamHandle stream_handle)
{
    RETURN_ON_INVALID_DATA_TYPE(datatype);

    auto* hccl_comm = hccl_ctx.communicator(comm);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);
    HCCL_CHECK_STOP_COLL_API_COMM_UNTIL(hccl_comm);

    hccl_comm->incCollectiveCtr();

    HCL_API_LOG_ENTRY("rank={}/{}, oam={}, (sendbuff={:p}, recvbuff={:p}, count={}, datatype={}, reduceOp={}, "
                      "root={}, uniqId={}, stream_handle={:p}) - collective#=0x{:x}",
                      hccl_comm->user_rank(),
                      hccl_comm->getCommSize(),
                      hccl_device()->getHwModuleId(),
                      (void*)sendbuff,
                      (void*)recvbuff,
                      count,
                      to_string(datatype),
                      to_string(reduceOp),
                      root,
                      hccl_comm->getCommUniqueId(),
                      (void*)stream_handle,
                      hccl_comm->getCollectiveCtr());

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    LOG_SYNC_DBG(Reduce,
                 "#Lines: 1 {:#x} {:#x} {:#x} {:#x} {:#x}",
                 TO64(stream_handle),
                 TO64(hccl_comm),
                 TO64(sendbuff),
                 TO64(recvbuff),
                 count * hccl_data_type_elem_size(datatype));

    return (*functions_pointers_table
                 ->pfn_hcclReduce)(sendbuff, recvbuff, count, datatype, reduceOp, root, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclBcast_impl(void*           buff,
                                          size_t          count,
                                          hcclDataType_t  datatype,
                                          int             root,
                                          hcclComm_t      comm,
                                          synStreamHandle stream_handle)
{
    auto* hccl_comm = hccl_ctx.communicator(comm);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);
    HCCL_CHECK_STOP_COLL_API_COMM_UNTIL(hccl_comm);

    hccl_comm->incCollectiveCtr();

    HCL_API_LOG_ENTRY("rank={}/{}, oam={}, (buff={:p}, count={}, datatype={}, root={}, uniqId={}, "
                      "stream_handle={:p}) - collective#=0x{:x}",
                      hccl_comm->user_rank(),
                      hccl_comm->getCommSize(),
                      hccl_device()->getHwModuleId(),
                      (void*)buff,
                      count,
                      to_string(datatype),
                      root,
                      hccl_comm->getCommUniqueId(),
                      (void*)stream_handle,
                      hccl_comm->getCollectiveCtr());

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    return (*functions_pointers_table->pfn_hcclBcast)(buff, count, datatype, root, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclBroadcast_impl(const void*     sendbuff,
                                              void*           recvbuff,
                                              size_t          count,
                                              hcclDataType_t  datatype,
                                              int             root,
                                              hcclComm_t      comm_handle,
                                              synStreamHandle stream_handle)
{
    RETURN_ON_INVALID_DATA_TYPE(datatype);

    auto* hccl_comm = hccl_ctx.communicator(comm_handle);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);
    HCCL_CHECK_STOP_COLL_API_COMM_UNTIL(hccl_comm);

    hccl_comm->incCollectiveCtr();

    HCL_API_LOG_ENTRY("rank={}/{}, oam={}, (sendbuff={:p}, recvbuff={:p}, count={}, datatype={}, root={}, "
                      "uniqId={}, stream_handle={:p}) - collective#=0x{:x}",
                      hccl_comm->user_rank(),
                      hccl_comm->getCommSize(),
                      hccl_device()->getHwModuleId(),
                      (void*)sendbuff,
                      (void*)recvbuff,
                      count,
                      to_string(datatype),
                      root,
                      hccl_comm->getCommUniqueId(),
                      (void*)stream_handle,
                      hccl_comm->getCollectiveCtr());

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    LOG_SYNC_DBG(Broadcast,
                 "#Lines: 1 {:#x} {:#x} {:#x} {:#x} {:#x}",
                 TO64(stream_handle),
                 TO64(hccl_comm),
                 TO64(sendbuff),
                 TO64(recvbuff),
                 count * hccl_data_type_elem_size(datatype));

    return (*functions_pointers_table
                 ->pfn_hcclBroadcast)(sendbuff, recvbuff, count, datatype, root, comm_handle, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclAllGather_impl(const void*     sendbuff,
                                              void*           recvbuff,
                                              size_t          sendcount,
                                              hcclDataType_t  datatype,
                                              hcclComm_t      comm_handle,
                                              synStreamHandle stream_handle)
{
    RETURN_ON_INVALID_DATA_TYPE(datatype);

    auto* hccl_comm = hccl_ctx.communicator(comm_handle);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);
    HCCL_CHECK_STOP_COLL_API_COMM_UNTIL(hccl_comm);

    hccl_comm->incCollectiveCtr();

    HCL_API_LOG_ENTRY("rank={}/{}, oam={}, (sendbuff={:p}, recvbuff={:p}, sendcount={}, datatype={}, uniqId={}, "
                      "stream_handle={:p}) - collective#=0x{:x}",
                      hccl_comm->user_rank(),
                      hccl_comm->getCommSize(),
                      hccl_device()->getHwModuleId(),
                      (void*)sendbuff,
                      (void*)recvbuff,
                      sendcount,
                      to_string(datatype),
                      hccl_comm->getCommUniqueId(),
                      (void*)stream_handle,
                      hccl_comm->getCollectiveCtr());

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    LOG_SYNC_DBG(AllGather,
                 "#Lines: 1 {:#x} {:#x} {:#x} {:#x} {:#x}",
                 TO64(stream_handle),
                 TO64(hccl_comm),
                 TO64(sendbuff),
                 TO64(recvbuff),
                 sendcount * hccl_data_type_elem_size(datatype));

    return (*functions_pointers_table
                 ->pfn_hcclAllGather)(sendbuff, recvbuff, sendcount, datatype, comm_handle, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclBarrier_impl([[maybe_unused]] hcclComm_t      comm_handle,
                                            [[maybe_unused]] synStreamHandle stream_handle)
{
    HCCL_TRY
    LOG_ERR(HCL_API, "Unsupported API for Gen2");
    HCCL_API_EXIT(hcclUnsupported)
}

hcclResult_t HCCL_API_CALL hcclAlltoAll_impl(const void*     sendbuff,
                                             void*           recvbuff,
                                             size_t          count,
                                             hcclDataType_t  datatype,
                                             hcclComm_t      comm,
                                             synStreamHandle stream_handle)
{
    RETURN_ON_INVALID_DATA_TYPE(datatype);

    auto* hccl_comm = hccl_ctx.communicator(comm);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);
    HCCL_CHECK_STOP_COLL_API_COMM_UNTIL(hccl_comm);

    hccl_comm->incCollectiveCtr();

    HCL_API_LOG_ENTRY("rank={}/{}, oam={}, (sendbuff={:p}, recvbuff={:p}, count={}, datatype={}, uniqId={}, "
                      "stream_handle={:p}) - collective#=0x{:x}",
                      hccl_comm->user_rank(),
                      hccl_comm->getCommSize(),
                      hccl_device()->getHwModuleId(),
                      (void*)sendbuff,
                      (void*)recvbuff,
                      count,
                      to_string(datatype),
                      hccl_comm->getCommUniqueId(),
                      (void*)stream_handle,
                      hccl_comm->getCollectiveCtr());

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    LOG_SYNC_DBG(AllToAll,
                 "#Lines: 1 {:#x} {:#x} {:#x} {:#x} {:#x}",
                 TO64(stream_handle),
                 TO64(hccl_comm),
                 TO64(sendbuff),
                 TO64(recvbuff),
                 count * hccl_data_type_elem_size(datatype));

    return (*functions_pointers_table->pfn_hcclAlltoAll)(sendbuff, recvbuff, count, datatype, comm, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclSend_impl(const void*     sendbuff,
                                         size_t          count,
                                         hcclDataType_t  datatype,
                                         int             peer,
                                         hcclComm_t      comm_handle,
                                         synStreamHandle stream_handle)
{
    RETURN_ON_INVALID_DATA_TYPE(datatype);

    auto* hccl_comm = hccl_ctx.communicator(comm_handle);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);

    const uint64_t send_cntr = hccl_comm->incSendCtr((HCL_Rank)peer);
    HCL_API_LOG_ENTRY(
        "rank={}/{}, oam={}, (sendbuff={:p}, count={}, datatype={}, uniqId={}, stream_handle={:p}, peer={}, send#={})",
        hccl_comm->user_rank(),
        hccl_comm->getCommSize(),
        hccl_device()->getHwModuleId(),
        (void*)sendbuff,
        count,
        to_string(datatype),
        hccl_comm->getCommUniqueId(),
        (void*)stream_handle,
        peer,
        send_cntr);

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    LOG_SYNC_DBG(Send,
                 "#Lines: 1 {:#x} {:#x} {:#x} {:#x}",
                 TO64(stream_handle),
                 TO64(hccl_comm),
                 TO64(sendbuff),
                 count * hccl_data_type_elem_size(datatype));

    return (*functions_pointers_table->pfn_hcclSend)(sendbuff, count, datatype, peer, comm_handle, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclRecv_impl(void*           recvbuff,
                                         size_t          count,
                                         hcclDataType_t  datatype,
                                         int             peer,
                                         hcclComm_t      comm_handle,
                                         synStreamHandle stream_handle)
{
    RETURN_ON_INVALID_DATA_TYPE(datatype);

    auto* hccl_comm = hccl_ctx.communicator(comm_handle);
    RETURN_ON_INVALID_HCCL_COMM(hccl_comm);

    const uint64_t recv_cntr = hccl_comm->incRecvCtr((HCL_Rank)peer);
    HCL_API_LOG_ENTRY(
        "rank={}/{}, oam={}, (recvbuff={:p}, count={}, datatype={}, uniqId={}, stream_handle={:p}, peer={}, recv#={})",
        hccl_comm->user_rank(),
        hccl_comm->getCommSize(),
        hccl_device()->getHwModuleId(),
        recvbuff,
        count,
        to_string(datatype),
        hccl_comm->getCommUniqueId(),
        (void*)stream_handle,
        peer,
        recv_cntr);

    hcclResult_t status = syncHCLStreamHandle(stream_handle);
    if (status != hcclSuccess) return status;

    LOG_SYNC_DBG(Receive,
                 "#Lines: 1 {:#x} {:#x} {:#x} {:#x}",
                 TO64(stream_handle),
                 TO64(hccl_comm),
                 TO64(recvbuff),
                 count * hccl_data_type_elem_size(datatype));

    return (*functions_pointers_table->pfn_hcclRecv)(recvbuff, count, datatype, peer, comm_handle, stream_handle);
}

hcclResult_t HCCL_API_CALL hcclGroupStart_impl()
{
    HCL_API_LOG_ENTRY();

    LOG_SYNC_DBG(GroupStart, "#Lines: 1");

    return (*functions_pointers_table->pfn_hcclGroupStart)();
}

hcclResult_t HCCL_API_CALL hcclGroupEnd_impl()
{
    HCL_API_LOG_ENTRY();

    LOG_SYNC_DBG(GroupEnd, "#Lines: 1");

    return (*functions_pointers_table->pfn_hcclGroupEnd)();
}

hcclResult_t HCCL_API_CALL hcclInitDevice(const synDeviceId deviceId)
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclInitDevice)(deviceId);
}

hcclResult_t HCCL_API_CALL hcclDestroyDevice(const synDeviceId deviceid)
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclDestroyDevice)(deviceid);
}

hcclResult_t HCCL_API_CALL hcclEventRecord(hcclEventHandle* eventHandle, synStreamHandle streamHandle)
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclEventRecord)(eventHandle, streamHandle);
}

hcclResult_t HCCL_API_CALL hcclSynchronizeEvent(const hcclEventHandle& eventHandle, uint64_t microSeconds)
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclSynchronizeEvent)(eventHandle, microSeconds);
}

hcclResult_t HCCL_API_CALL hcclFlushSubmissions(synStreamHandle streamHandle)
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclFlushSubmissions)(streamHandle);
}

hcclResult_t HCCL_API_CALL hcclFlushSubmissionsAllStreams()
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclFlushSubmissionsAllStreams)();
}

hcclResult_t HCCL_API_CALL hcclSynchronizeStream(synStreamHandle streamHandle)
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclSynchronizeStream)(streamHandle);
}

hcclResult_t HCCL_API_CALL hcclSynchronizeAllStreams()
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclSynchronizeAllStreams)();
}

hcclResult_t HCCL_API_CALL hcclDFA(DfaStatus& dfaStatus, void (*dfaLogFunc)(int, const char*))
{
    HCL_API_LOG_ENTRY();
    return (*functions_pointers_table->pfn_hcclDFA)(dfaStatus, dfaLogFunc);
}

hcclResult_t HCCL_API_CALL hcclDfaUpdateState(DfaPhase dfaPhase)
{
    HCL_API_LOG_ENTRY("update Dfa phase to {}", dfaPhase);
    return (*functions_pointers_table->pfn_hcclDfaUpdateState)(dfaPhase);
}

hcclResult_t HCCL_API_CALL hcclGetVersionString(char* pVersion, const unsigned len)
{
    HCL_API_LOG_ENTRY("pVersion={:#x}, len={}", TO64(pVersion), len);
    return (*functions_pointers_table->pfn_hcclGetVersionString)(pVersion, len);
}

hcclResult_t HCCL_API_CALL hcclDeviceInit_impl(void* device, void* context)
{
    HCL_API_LOG_ENTRY("(&device={:p}, &context={:p})", device, context);
    return (*functions_pointers_table->pfn_hcclDeviceInit)(device, context);
}

}  // namespace HclGen2

extern "C" hcclResult_t HCCL_API_CALL hcclTp2DirectAllReduce(const void*     sendbuff,
                                                              void*           recvbuff,
                                                              size_t          count,
                                                              hcclDataType_t  datatype,
                                                              hcclRedOp_t     reduceOp,
                                                              hcclComm_t      comm,
                                                              synStreamHandle stream_handle)
{
    return HclGen2::hcclTp2DirectAllReduce_impl(
        sendbuff, recvbuff, count, datatype, reduceOp, comm, stream_handle);
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2DirectExchange(const void*     sendbuff,
                                                            void*           recvbuff,
                                                            size_t          count,
                                                            hcclDataType_t  datatype,
                                                            hcclRedOp_t     reduceOp,
                                                            hcclComm_t      comm,
                                                            synStreamHandle stream_handle)
{
    return HclGen2::hcclTp2DirectExchange_impl(
        sendbuff, recvbuff, count, datatype, reduceOp, comm, stream_handle);
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphCreate(
    const void*               sendbuff,
    void*                     recvbuff,
    size_t                    count,
    hcclDataType_t            datatype,
    hcclRedOp_t               reduceOp,
    hcclComm_t                comm,
    void*                     streamHandle,
    hcclTp2NativeMode         mode,
    hcclTp2NativeGraphHandle* graphHandle)
{
    if (graphHandle == nullptr || sendbuff == nullptr || recvbuff == nullptr || sendbuff == recvbuff || count == 0 ||
        datatype != hcclBfloat16 || reduceOp != hcclSum ||
        (mode != hcclTp2NativeReduction && mode != hcclTp2NativeExchange) || streamHandle == nullptr ||
        !GCFG_HCL_TP2_DEDICATED_DIRECT_API.value())
    {
        return hcclInvalidArgument;
    }
    *graphHandle = nullptr;

    const uint64_t maxCount = GCFG_HCL_TP2_DIRECT_MAX_COUNT.value();
    if (maxCount > 0 && count > maxCount)
    {
        return hcclInvalidArgument;
    }

    auto* communicator = hccl_ctx.communicator(comm);
    if (communicator == nullptr || communicator->getDynamicComm() == nullptr || communicator->getCommSize() != 2 ||
        communicator->getDynamicComm()->getScaleupGroupSize() != 2)
    {
        return hcclInvalidArgument;
    }

    try
    {
        const uint32_t streamId = hccl_device().nativeStreamId(streamHandle);
        HclCollectiveRoutinesGen2Arch* routines = hccl_device().nativeCollectiveRoutines(streamId);
        if (routines == nullptr)
        {
            return hcclInvalidUsage;
        }

        auto* handle = new InternalHcclTp2NativeGraph(communicator,
                                                       routines,
                                                       *communicator->getDynamicComm(),
                                                       sendbuff,
                                                       recvbuff,
                                                       count,
                                                       datatype,
                                                       reduceOp,
                                                       streamHandle,
                                                       mode,
                                                       hccl_ctx.generateApiId());
        handle->graph = ar_createTp2DirectGraph(
            routines, handle->params, mode == hcclTp2NativeReduction);
        if (handle->graph == nullptr || !handle->graph->instantiated())
        {
            delete handle;
            return hcclInternalError;
        }
        *graphHandle = handle;
        return hcclSuccess;
    }
    catch (...)
    {
        return hcclInternalError;
    }
}

// Capability is an artifact property, not a serving optimization switch.
extern "C" uint32_t HCCL_API_CALL hcclTp4NativeGraphVersion() { return 1; }

extern "C" hcclResult_t HCCL_API_CALL hcclTp4NativeGraphCreate(
    const void* sendbuff, void* recvbuff, size_t count, hcclDataType_t datatype,
    hcclRedOp_t reduceOp, hcclComm_t comm, void* streamHandle, int phase,
    hcclTp2NativeGraphHandle* graphHandle)
{
    if (!graphHandle || !sendbuff || !recvbuff || !streamHandle || datatype != hcclBfloat16 ||
        reduceOp != hcclSum || count == 0 || count > 32768 || phase < 0 || phase > 2 ||
        (phase != 2 && count % 4 != 0) || (phase == 1 ? sendbuff != recvbuff : sendbuff == recvbuff))
        return hcclInvalidArgument;
    *graphHandle = nullptr;
    auto* communicator = hccl_ctx.communicator(comm);
    if (!communicator || !communicator->getDynamicComm() || communicator->getCommSize() != 4 ||
        communicator->getDynamicComm()->getScaleupGroupSize() != 4) return hcclInvalidArgument;
    try
    {
        const uint32_t streamId = hccl_device().nativeStreamId(streamHandle);
        auto* routines = hccl_device().nativeCollectiveRoutines(streamId);
        if (!routines) return hcclInvalidUsage;
        auto handle = std::make_unique<InternalHcclTp2NativeGraph>(communicator, routines,
            *communicator->getDynamicComm(), sendbuff, recvbuff, count, datatype, reduceOp,
            streamHandle, hcclTp2NativeReduction, hccl_ctx.generateApiId());
        handle->tp4Phase = phase;
        handle->graph = ar_createTp4NativePhaseGraph(routines, handle->params, phase);
        if (!handle->graph || !handle->graph->instantiated()) return hcclInternalError;
        *graphHandle = handle.release();
        return hcclSuccess;
    }
    catch (...) { return hcclInternalError; }
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphCapture(hcclTp2NativeGraphHandle graphHandle)
{
    if (graphHandle == nullptr)
    {
        return hcclInvalidArgument;
    }

    std::lock_guard<std::mutex> lock(graphHandle->mutex);
    if (graphHandle->state != hcclTp2NativeGraphCreated)
    {
        return hcclInvalidUsage;
    }

    const hcclResult_t syncStatus =
        syncHCLStreamHandle(reinterpret_cast<synStreamHandle>(graphHandle->streamHandle));
    if (syncStatus != hcclSuccess)
    {
        graphHandle->state = hcclTp2NativeGraphInvalid;
        return syncStatus;
    }

    auto reject = [&](const char* reason) {
        std::fprintf(stderr, "NATIVE_HCL_CAPTURE_RETURN phase=%d count=%zu reason=%s\n",
                     graphHandle->tp4Phase, size_t(graphHandle->params.m_count), reason);
        graphHandle->state = hcclTp2NativeGraphInvalid;
        return hcclInternalError;
    };
    bool forcedBufferAddressRefresh = false;
    try
    {
        // The first execution installs NIC collective context and emits two
        // one-time scheduler commands.  Those commands are valid cold-path
        // setup, but they are not a replayable steady-state template.
        graphHandle->communicator->incCollectiveCtr();
        graphHandle->params.m_apiId = hccl_ctx.generateApiId();
        const hcclResult_t warmupStatus = graphHandle->graph->submit();
        if (warmupStatus != hcclSuccess)
        {
            graphHandle->state = hcclTp2NativeGraphInvalid;
            std::fprintf(stderr, "NATIVE_HCL_SUBMIT phase=%d count=%zu stage=warmup status=%d\n", graphHandle->tp4Phase, size_t(graphHandle->params.m_count), int(warmupStatus));
            return warmupStatus;
        }
        // A second Synapse stream sync does not create a new dependency when
        // no compute was added to the stream.  Use the completed cold HCL
        // epoch as the capture-time placeholder instead.  Its long-SO fields
        // are relocation sites and are replaced with the current compute
        // completion on every native replay.
        graphHandle->routines->nativeWaitOn(graphHandle->routines->nativeCompletion());

        graphHandle->communicator->incCollectiveCtr();
        graphHandle->params.m_apiId = hccl_ctx.generateApiId();
        graphHandle->state = hcclTp2NativeGraphCapturing;
        // Each fixed-buffer template installs its own send/receive address
        // MSB. Keep synchronization fields on their real completion phase.
        graphHandle->routines->setNativeScaleupBufferAddressRefresh(true);
        forcedBufferAddressRefresh = true;
        graphHandle->routines->beginNativeCapture();
        const hcclResult_t status = graphHandle->graph->submit();
        graphHandle->routines->setNativeScaleupBufferAddressRefresh(false);
        forcedBufferAddressRefresh = false;
        if (status != hcclSuccess)
        {
            graphHandle->routines->cancelNativeCapture();
            graphHandle->state = hcclTp2NativeGraphInvalid;
            std::fprintf(stderr, "NATIVE_HCL_SUBMIT phase=%d count=%zu stage=capture_submit status=%d\n", graphHandle->tp4Phase, size_t(graphHandle->params.m_count), int(status));
            return status;
        }
        graphHandle->capture = graphHandle->routines->endNativeCapture();
        graphHandle->capture.tp4Phase = graphHandle->tp4Phase;
        if (graphHandle->capture.commandCount() == 0)
        {
            return reject("empty_commands");
        }

        // Reusing the first completion-SO slot requires a transient NIC
        // collective-context update on both scale-up streams.  Capture that
        // boundary shape once on the cold path as a second native template.
        // Intermediate executions are ordinary instantiation work; replay
        // never calls HcclGraph::submit().
        graphHandle->completionRingSize = graphHandle->routines->nativeCompletionRingSize();
        const uint64_t currentTarget = graphHandle->routines->nativeCompletion().targetValue;
        if (graphHandle->completionRingSize < 2 || currentTarget == 0)
        {
            return reject("completion_ring_or_target");
        }
        const uint64_t position = (currentTarget - 1) % graphHandle->completionRingSize;
        const uint64_t executionsToWrap = graphHandle->completionRingSize - position;
        for (uint64_t execution = 1; execution <= executionsToWrap; ++execution)
        {
            graphHandle->routines->nativeWaitOn(graphHandle->routines->nativeCompletion());
            const bool captureWrap = execution == executionsToWrap;
            if (captureWrap)
            {
                graphHandle->routines->setNativeScaleupBufferAddressRefresh(true);
                forcedBufferAddressRefresh = true;
                graphHandle->routines->beginNativeCapture();
            }
            graphHandle->communicator->incCollectiveCtr();
            graphHandle->params.m_apiId = hccl_ctx.generateApiId();
            const hcclResult_t wrapStatus = graphHandle->graph->submit();
            if (captureWrap)
            {
                graphHandle->routines->setNativeScaleupBufferAddressRefresh(false);
                forcedBufferAddressRefresh = false;
            }
            if (wrapStatus != hcclSuccess)
            {
                if (captureWrap)
                {
                    graphHandle->routines->cancelNativeCapture();
                }
                graphHandle->state = hcclTp2NativeGraphInvalid;
                std::fprintf(stderr, "NATIVE_HCL_SUBMIT phase=%d count=%zu stage=wrap_submit status=%d\n", graphHandle->tp4Phase, size_t(graphHandle->params.m_count), int(wrapStatus));
            return wrapStatus;
            }
            if (captureWrap)
            {
                graphHandle->wrapCapture = graphHandle->routines->endNativeCapture();
                graphHandle->wrapCapture.tp4Phase = graphHandle->tp4Phase;
            }
        }
        if (!graphHandle->wrapCapture.validate() ||
            graphHandle->wrapCapture.commandCount() == 0 ||
            (graphHandle->tp4Phase < 0 &&
             graphHandle->wrapCapture.commandCount() < graphHandle->capture.commandCount()) ||
            graphHandle->wrapCapture.externalTargetDelta != graphHandle->capture.externalTargetDelta ||
            graphHandle->wrapCapture.internalTargetDelta != graphHandle->capture.internalTargetDelta)
        {
            std::fprintf(stderr, "NATIVE_HCL_WRAP valid=%d commands=%zu/%zu delta=%llu/%llu internal=%llu/%llu\n",
                int(graphHandle->wrapCapture.validate()),
                graphHandle->capture.commandCount(), graphHandle->wrapCapture.commandCount(),
                static_cast<unsigned long long>(graphHandle->capture.externalTargetDelta),
                static_cast<unsigned long long>(graphHandle->wrapCapture.externalTargetDelta),
                static_cast<unsigned long long>(graphHandle->capture.internalTargetDelta),
                static_cast<unsigned long long>(graphHandle->wrapCapture.internalTargetDelta));
            return reject("wrap_contract");
        }

        // Every TP4 physical phase owns two independently validated complete
        // templates. Context elision may make its wrap template shorter; an
        // opcode-count ordering is not a dependency/completion invariant.
        // Keep each template's command/relocation validation and both target
        // deltas, and retain the historical check for TP2.
        if (graphHandle->tp4Phase >= 0 &&
            graphHandle->wrapCapture.commandCount() < graphHandle->capture.commandCount())
        {
            dumpNativeCaptureDiff(graphHandle->capture, graphHandle->wrapCapture, 0, executionsToWrap);
        }

        for (const auto* candidate : {&graphHandle->capture, &graphHandle->wrapCapture})
        {
            const size_t targets = candidate->relocationCount(hcl::NativeHclRelocationKind::DEPENDENCY_TARGET);
            const size_t indexes = candidate->relocationCount(hcl::NativeHclRelocationKind::DEPENDENCY_LONG_SO_INDEX);
            const size_t notify = candidate->relocationCount(hcl::NativeHclRelocationKind::WQE_NOTIFY);
            const size_t wait = candidate->relocationCount(hcl::NativeHclRelocationKind::WQE_WAIT);
            if (!candidate->executableValidated || targets == 0 || targets % 4 != 0 ||
                indexes * 4 != targets || notify == 0 || notify != wait)
            {
                std::fprintf(stderr, "NATIVE_HCL_FIELDS validated=%d targets=%zu indexes=%zu notify=%zu wait=%zu\n",
                             int(candidate->executableValidated), targets, indexes, notify, wait);
                return reject("dependency_relocations");
            }
        }
        graphHandle->nativeRelocationsValidated = true;
        graphHandle->captureCounters = graphHandle->routines->nativeReplayCounters(graphHandle->capture);
        const char* diagnostics = std::getenv("HCL_TP2_NATIVE_REPLAY_DIAGNOSTICS");
        if (diagnostics != nullptr && std::strcmp(diagnostics, "1") == 0)
        {
            std::fprintf(stderr,
                         "NATIVE_HCL_CAPTURE commands=%zu/%zu streams=%zu ring=%llu dependency=%zu/%zu "
                         "wqe=%zu/%zu completion_addresses=%zu/%zu\n",
                         graphHandle->capture.commandCount(),
                         graphHandle->wrapCapture.commandCount(),
                         graphHandle->capture.streams.size(),
                         static_cast<unsigned long long>(graphHandle->completionRingSize),
                         graphHandle->capture.relocationCount(hcl::NativeHclRelocationKind::DEPENDENCY_TARGET),
                         graphHandle->capture.relocationCount(
                             hcl::NativeHclRelocationKind::DEPENDENCY_LONG_SO_INDEX),
                         graphHandle->capture.relocationCount(hcl::NativeHclRelocationKind::WQE_NOTIFY),
                         graphHandle->capture.relocationCount(hcl::NativeHclRelocationKind::WQE_WAIT),
                         graphHandle->capture.relocationCount(
                             hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS),
                         graphHandle->capture.relocationCount(
                             hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS));
            std::fflush(stderr);
        }
        graphHandle->state = hcclTp2NativeGraphInstantiated;
        return hcclSuccess;
    }
    catch (...)
    {
        try { throw; }
        catch (const std::exception& error) {
            std::fprintf(stderr, "NATIVE_HCL_CAPTURE_REJECT phase=%d count=%zu reason=%s\n",
                         graphHandle->tp4Phase, size_t(graphHandle->params.m_count), error.what());
        } catch (...) {}
        if (forcedBufferAddressRefresh)
        {
            graphHandle->routines->setNativeScaleupBufferAddressRefresh(false);
        }
        graphHandle->routines->cancelNativeCapture();
        graphHandle->state = hcclTp2NativeGraphInvalid;
        return hcclInternalError;
    }
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphReplayPrepared(
    hcclTp2NativeGraphHandle   graphHandle,
    const hcclTp2NativeSyncInfo* dependency,
    hcclTp2NativeSyncInfo*       completion)
{
    if (graphHandle == nullptr || completion == nullptr)
    {
        return hcclInvalidArgument;
    }

    std::lock_guard<std::mutex> lock(graphHandle->mutex);
    if (graphHandle->batchOwned) return hcclInvalidUsage;
    if (graphHandle->state != hcclTp2NativeGraphInstantiated)
    {
        return hcclInvalidUsage;
    }
    if (graphHandle->replayPath == InternalHcclTp2NativeGraph::ReplayPath::Native)
    {
        return hcclInvalidUsage;
    }

    try
    {
        if (dependency != nullptr && dependency->targetValue != 0)
        {
            graphHandle->routines->nativeWaitOn(
                {.long_so_index = dependency->longSoIndex, .targetValue = dependency->targetValue});
        }
        graphHandle->communicator->incCollectiveCtr();
        graphHandle->params.m_apiId = hccl_ctx.generateApiId();
        const char* comparePrepared = std::getenv("HCL_TP2_NATIVE_COMPARE_PREPARED");
        const bool comparePreparedEnabled =
            comparePrepared != nullptr && std::strcmp(comparePrepared, "1") == 0;
        const bool capturePrepared = comparePreparedEnabled &&
                                     (graphHandle->preparedReplayCount < 3 ||
                                      (graphHandle->preparedReplayCount >= 60 &&
                                       graphHandle->preparedReplayCount < 65));
        if (capturePrepared)
        {
            graphHandle->routines->beginNativeCapture();
        }
        const hcclResult_t status = graphHandle->graph->submit();
        if (status != hcclSuccess)
        {
            if (capturePrepared)
            {
                graphHandle->routines->cancelNativeCapture();
            }
            graphHandle->state = hcclTp2NativeGraphInvalid;
            return status;
        }
        if (capturePrepared)
        {
            const auto preparedCapture = graphHandle->routines->endNativeCapture();
            const auto& previousCapture = graphHandle->diagnosticPreparedCaptureValid
                                              ? graphHandle->diagnosticPreparedCapture
                                              : graphHandle->capture;
            const uint64_t currentExecution = graphHandle->preparedReplayCount + 3;
            dumpNativeCaptureDiff(previousCapture,
                                  preparedCapture,
                                  graphHandle->diagnosticPreparedCaptureValid
                                      ? graphHandle->diagnosticPreparedExecution
                                      : 2,
                                  currentExecution);
            graphHandle->diagnosticPreparedCapture = preparedCapture;
            graphHandle->diagnosticPreparedCaptureValid = true;
            graphHandle->diagnosticPreparedExecution = currentExecution;
        }
        const hcl::syncInfo done = graphHandle->routines->nativeCompletion();
        completion->longSoIndex = done.long_so_index;
        completion->targetValue = done.targetValue;
        graphHandle->replayPath = InternalHcclTp2NativeGraph::ReplayPath::Prepared;
        ++graphHandle->preparedReplayCount;
        return hcclSuccess;
    }
    catch (...)
    {
        graphHandle->state = hcclTp2NativeGraphInvalid;
        return hcclInternalError;
    }
}

static hcclResult_t replayTp2NativeGraphLocked(hcclTp2NativeGraphHandle    graphHandle,
                                               const hcclTp2NativeSyncInfo* dependency,
                                               hcclTp2NativeSyncInfo*       completion,
                                               bool                         submit)
{
    if (graphHandle->state != hcclTp2NativeGraphInstantiated ||
        graphHandle->replayPath == InternalHcclTp2NativeGraph::ReplayPath::Prepared)
    {
        return hcclInvalidUsage;
    }

    const hcl::syncInfo currentCompletion = graphHandle->routines->nativeCompletion();
    const uint64_t nextTarget = currentCompletion.targetValue + graphHandle->capture.externalTargetDelta;
    const bool useWrapCapture = graphHandle->completionRingSize > 0 && nextTarget > 1 &&
                                (nextTarget - 1) % graphHandle->completionRingSize == 0;
    const auto& selectedCapture = useWrapCapture ? graphHandle->wrapCapture : graphHandle->capture;
    if (!graphHandle->nativeRelocationsValidated || graphHandle->batchOwned)
    {
        graphHandle->state = hcclTp2NativeGraphInvalid;
        return hcclInvalidUsage;
    }

    try
    {
        graphHandle->communicator->incCollectiveCtr();
        const hcl::syncInfo done = graphHandle->routines->replayNativeCapture(
            selectedCapture,
            *graphHandle->communicator->getDynamicComm(),
            {.long_so_index = dependency->longSoIndex, .targetValue = dependency->targetValue},
            submit);
        completion->longSoIndex = done.long_so_index;
        completion->targetValue = done.targetValue;
        graphHandle->replayPath = InternalHcclTp2NativeGraph::ReplayPath::Native;
        ++graphHandle->nativeReplayCount;
        graphHandle->nativeWrapReplayCount += useWrapCapture;
        return hcclSuccess;
    }
    catch (...)
    {
        graphHandle->state = hcclTp2NativeGraphInvalid;
        return hcclInternalError;
    }
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphReplay(
    hcclTp2NativeGraphHandle    graphHandle,
    const hcclTp2NativeSyncInfo* dependency,
    hcclTp2NativeSyncInfo*       completion)
{
    if (graphHandle == nullptr || dependency == nullptr || dependency->targetValue == 0 || completion == nullptr)
    {
        return hcclInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(graphHandle->mutex);
    return replayTp2NativeGraphLocked(graphHandle, dependency, completion, true);
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphStage(
    hcclTp2NativeGraphHandle    graphHandle,
    const hcclTp2NativeSyncInfo* dependency,
    hcclTp2NativeSyncInfo*       completion)
{
    if (graphHandle == nullptr || dependency == nullptr || dependency->targetValue == 0 || completion == nullptr)
    {
        return hcclInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(graphHandle->mutex);
    return replayTp2NativeGraphLocked(graphHandle, dependency, completion, false);
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphSubmit(hcclTp2NativeGraphHandle graphHandle)
{
    if (graphHandle == nullptr)
    {
        return hcclInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(graphHandle->mutex);
    if (graphHandle->state != hcclTp2NativeGraphInstantiated ||
        graphHandle->replayPath != InternalHcclTp2NativeGraph::ReplayPath::Native)
    {
        return hcclInvalidUsage;
    }
    try
    {
        graphHandle->routines->submitNativeReplay();
        return hcclSuccess;
    }
    catch (...)
    {
        graphHandle->state = hcclTp2NativeGraphInvalid;
        return hcclInternalError;
    }
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphGetInfo(hcclTp2NativeGraphHandle graphHandle,
                                                                 hcclTp2NativeGraphInfo*   graphInfo)
{
    if (graphHandle == nullptr || graphInfo == nullptr)
    {
        return hcclInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(graphHandle->mutex);
    graphInfo->state                = graphHandle->state;
    graphInfo->preparedReplayCount  = graphHandle->preparedReplayCount;
    graphInfo->nativeReplayCount    = graphHandle->nativeReplayCount;
    graphInfo->capturedCommandCount = graphHandle->capture.commandCount();
    graphInfo->capturedStreamCount  = graphHandle->capture.streams.size();
    graphInfo->capturedRelocationCount = 0;
    graphInfo->capturedByteCount = 0;
    graphInfo->maxStreamBytes = 0;
    for (const auto& stream : graphHandle->capture.streams)
    {
        graphInfo->capturedRelocationCount += stream.commandTemplate.relocations.size();
        uint64_t streamBytes = 0;
        for (const auto& command : stream.commandTemplate.commands)
        {
            streamBytes += command.bytes.size();
        }
        graphInfo->capturedByteCount += streamBytes;
        graphInfo->maxStreamBytes = std::max(graphInfo->maxStreamBytes, streamBytes);
    }
    graphInfo->streamCcbBytes = hcl::ScalStream::getCcbSize();
    const auto replayCounters = graphHandle->routines->nativeReplayCounters(graphHandle->capture);
    graphInfo->nativeReplayBytes = replayCounters.bytes >= graphHandle->captureCounters.bytes
                                       ? replayCounters.bytes - graphHandle->captureCounters.bytes
                                       : 0;
    graphInfo->nativeCcbWrapCount = replayCounters.ccbWraps >= graphHandle->captureCounters.ccbWraps
                                        ? replayCounters.ccbWraps - graphHandle->captureCounters.ccbWraps
                                        : 0;
    graphInfo->nativeSubmissionCount = replayCounters.submissions >= graphHandle->captureCounters.submissions
                                           ? replayCounters.submissions - graphHandle->captureCounters.submissions
                                           : 0;
    graphInfo->wrapCapturedCommandCount = graphHandle->wrapCapture.commandCount();
    graphInfo->completionRingSize = graphHandle->completionRingSize;
    graphInfo->nativeWrapReplayCount = graphHandle->nativeWrapReplayCount;
    graphInfo->dependencyRelocationCount = graphHandle->capture.relocationCount(
        hcl::NativeHclRelocationKind::DEPENDENCY_TARGET);
    graphInfo->wqeRelocationCount =
        graphHandle->capture.relocationCount(hcl::NativeHclRelocationKind::WQE_NOTIFY) +
        graphHandle->capture.relocationCount(hcl::NativeHclRelocationKind::WQE_WAIT);
    graphInfo->externalCompletionAddressRelocationCount = graphHandle->capture.relocationCount(
        hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS);
    graphInfo->internalCompletionAddressRelocationCount = graphHandle->capture.relocationCount(
        hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS);
    const hcl::syncInfo done         = graphHandle->routines->nativeCompletion();
    graphInfo->completion.longSoIndex = done.long_so_index;
    graphInfo->completion.targetValue = done.targetValue;
    return hcclSuccess;
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeGraphDestroy(hcclTp2NativeGraphHandle graphHandle)
{
    if (graphHandle == nullptr)
    {
        return hcclInvalidArgument;
    }

    std::unique_lock<std::mutex> lock(graphHandle->mutex);
    if (graphHandle->batchOwned) return hcclInvalidUsage;
    try
    {
        if (graphHandle->state == hcclTp2NativeGraphCapturing)
        {
            graphHandle->routines->cancelNativeCapture();
        }
        graphHandle->routines->nativeSynchronize();
        graphHandle->state = hcclTp2NativeGraphClosed;
    }
    catch (...)
    {
        graphHandle->state = hcclTp2NativeGraphInvalid;
        return hcclInternalError;
    }
    lock.unlock();
    delete graphHandle;
    return hcclSuccess;
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeBatchCreate(
    const hcclTp2NativeGraphHandle* graphs, size_t count, hcclTp2NativeBatchHandle* result)
{
    if (!graphs || !result || count == 0 || count > 256) return hcclInvalidArgument;
    *result = nullptr;
    auto batch = std::make_unique<InternalHcclTp2NativeBatch>();
    // Roll back leases on preparation failure; no commands have been submitted.
    try
    {
        for (size_t index = 0; index < count; ++index)
        {
            auto graph = graphs[index];
            VERIFY(graph != nullptr, "Null native batch graph");
            std::lock_guard<std::mutex> lock(graph->mutex);
            VERIFY(!graph->batchOwned && graph->nativeRelocationsValidated &&
                   graph->state == hcclTp2NativeGraphInstantiated && graph->capture.externalTargetDelta == 1 &&
                   graph->wrapCapture.externalTargetDelta == 1 && graph->completionRingSize == 64,
                   "Native batch requires exclusive, qualified, unit-completion TP2 graphs");
            if (index != 0)
                VERIFY(graph->communicator == graphs[0]->communicator && graph->routines == graphs[0]->routines &&
                       graph->streamHandle == graphs[0]->streamHandle,
                       "Native batch topology or communicator differs");
            if (graph->tp4Phase == 1)
            {
                VERIFY(index > 0 && graphs[index - 1]->tp4Phase == 0 &&
                       graphs[index - 1]->params.m_count == graph->params.m_count &&
                       graphs[index - 1]->params.m_recvBufferAddr == graph->params.m_sendBufferAddr &&
                       graph->params.m_sendBufferAddr == graph->params.m_recvBufferAddr,
                       "TP4 AllGather requires its immediately preceding ReduceScatter output");
            }
            batch->graphs.push_back(graph);
            graph->batchOwned = true;
        }
        batch->states.resize(count);
        batch->dependencies.resize(count);
        batch->completions.resize(count);
        batch->phases = prepareNativeBatchPhases(*batch, 0, count);
        *result = batch.release();
        return hcclSuccess;
    }
    catch (...)
    {
        for (auto graph : batch->graphs)
        {
            std::lock_guard<std::mutex> lock(graph->mutex);
            graph->batchOwned = false;
        }
        return hcclInvalidUsage;
    }
}

static hcclResult_t replayNativeBatchRangeLocked(
    InternalHcclTp2NativeBatch& batch, size_t offset,
    const InternalHcclTp2NativeBatch::PhaseSet& phases,
    const hcclTp2NativeSyncInfo* dependencies, size_t count,
    hcclTp2NativeSyncInfo* completions)
{
    if (!batch.valid || phases.size() != 64 || !dependencies || !completions || count == 0 ||
        offset > batch.graphs.size() || count > batch.graphs.size() - offset) return hcclInvalidArgument;
    // Validate all dynamic fields before advancing counters or mutating queues.
    for (size_t node = 0; node < count; ++node)
    {
        if (!dependencies[node].targetValue || dependencies[node].longSoIndex % 8 != 0 ||
            (dependencies[node].longSoIndex >> 3) > 0xff ||
            dependencies[node].targetValue >= (uint64_t(1) << 60)) return hcclInvalidArgument;
        batch.dependencies[offset + node] = {dependencies[node].longSoIndex, dependencies[node].targetValue};
    }
    auto first = batch.graphs.front();
    std::lock_guard<std::mutex> streamLock(first->routines->nativeStreamLock());
    const uint64_t target = first->routines->nativeCompletion().targetValue;
    if (first->routines->nativeCompletion().long_so_index > std::numeric_limits<uint32_t>::max() ||
        !target || target >= (uint64_t(1) << 60) - count) return hcclInvalidUsage;
    // AllGather must wait for the preceding ReduceScatter completion, not
    // merely its earlier compute producer. The existing unit-completion
    // contract makes that native HCL epoch known before packet publication.
    for (size_t node = 0; node < count; ++node)
    {
        if (batch.graphs[offset + node]->tp4Phase == 1)
        {
            if (node == 0 || batch.graphs[offset + node - 1]->tp4Phase != 0) return hcclInvalidArgument;
            batch.dependencies[offset + node] = {first->routines->nativeCompletion().long_so_index, target + node};
        }
    }
    const auto& phase = phases[target % phases.size()];
    try
    {
        for (size_t node = 0; node < count; ++node) first->communicator->incCollectiveCtr();
        first->routines->replayNativeBatch(phase.program, phase.captures,
            *first->communicator->getDynamicComm(), batch.dependencies.data() + offset,
            batch.states.data() + offset, batch.completions.data() + offset);
        for (size_t node = 0; node < count; ++node)
        {
            auto graph = batch.graphs[offset + node];
            std::lock_guard<std::mutex> graphLock(graph->mutex);
            ++graph->nativeReplayCount;
            graph->nativeWrapReplayCount += phase.captures[node] == &graph->wrapCapture;
            graph->replayPath = InternalHcclTp2NativeGraph::ReplayPath::Native;
            const auto& completion = batch.completions[offset + node];
            completions[node] = {static_cast<uint32_t>(completion.long_so_index), completion.targetValue};
        }
        return hcclSuccess;
    }
    catch (...)
    {
        batch.valid = false;
        for (auto graph : batch.graphs) graph->state = hcclTp2NativeGraphInvalid;
        return hcclInternalError;
    }
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeBatchReplay(
    hcclTp2NativeBatchHandle batch, const hcclTp2NativeSyncInfo* dependencies,
    size_t count, hcclTp2NativeSyncInfo* completions)
{
    if (!batch || count != batch->graphs.size()) return hcclInvalidArgument;
    std::lock_guard<std::mutex> lock(batch->mutex);
    if (batch->segmentedActive) return hcclInvalidUsage;
    return replayNativeBatchRangeLocked(*batch, 0, batch->phases, dependencies, count, completions);
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeBatchConfigureSplit(
    hcclTp2NativeBatchHandle batch, size_t prefixCount)
{
    if (!batch || prefixCount == 0 || prefixCount >= batch->graphs.size()) return hcclInvalidArgument;
    std::lock_guard<std::mutex> lock(batch->mutex);
    if (!batch->valid || batch->split || batch->segmentedActive) return hcclInvalidUsage;
    try
    {
        batch->prefixPhases = prepareNativeBatchPhases(*batch, 0, prefixCount);
        batch->suffixPhases = prepareNativeBatchPhases(*batch, prefixCount, batch->graphs.size() - prefixCount);
        batch->split = prefixCount;
        return hcclSuccess;
    }
    catch (...)
    {
        batch->prefixPhases.clear();
        batch->suffixPhases.clear();
        batch->split = 0;
        return hcclInvalidUsage;
    }
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeBatchReplaySegmented(
    hcclTp2NativeBatchHandle batch, const hcclTp2NativeSyncInfo* dependencies,
    size_t count, hcclTp2NativeSyncInfo* completions)
{
    if (!batch || !dependencies || !completions) return hcclInvalidArgument;
    std::lock_guard<std::mutex> lock(batch->mutex);
    if (!batch->valid || !batch->split) return hcclInvalidUsage;
    if (!batch->segmentedActive)
    {
        if (count == batch->graphs.size())
            return replayNativeBatchRangeLocked(*batch, 0, batch->phases, dependencies, count, completions);
        if (count != batch->split) return hcclInvalidArgument;
        const hcclResult_t result = replayNativeBatchRangeLocked(
            *batch, 0, batch->prefixPhases, dependencies, count, completions);
        if (result == hcclSuccess) batch->segmentedActive = true;
        return result;
    }
    if (count != batch->graphs.size() - batch->split) return hcclInvalidArgument;
    const hcclResult_t result = replayNativeBatchRangeLocked(
        *batch, batch->split, batch->suffixPhases, dependencies, count, completions);
    if (result == hcclSuccess) batch->segmentedActive = false;
    return result;
}

extern "C" hcclResult_t HCCL_API_CALL hcclTp2NativeBatchDestroy(hcclTp2NativeBatchHandle batch)
{
    if (!batch) return hcclInvalidArgument;
    std::unique_lock<std::mutex> lock(batch->mutex);
    try
    {
        batch->graphs.front()->routines->nativeSynchronize();
        for (auto graph : batch->graphs)
        {
            std::lock_guard<std::mutex> graphLock(graph->mutex);
            graph->batchOwned = false;
        }
    }
    catch (...) { batch->valid = false; return hcclInternalError; }
    lock.unlock();
    delete batch;
    return hcclSuccess;
}
