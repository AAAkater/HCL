#pragma once

#include "hccl_types.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct InternalHcclTp2NativeGraph* hcclTp2NativeGraphHandle;
typedef struct InternalHcclTp2NativeBatch* hcclTp2NativeBatchHandle;

typedef enum
{
    hcclTp2NativeGraphCreated = 0,
    hcclTp2NativeGraphCapturing,
    hcclTp2NativeGraphInstantiated,
    hcclTp2NativeGraphInvalid,
    hcclTp2NativeGraphClosed
} hcclTp2NativeGraphState;

typedef enum
{
    hcclTp2NativeReduction = 0,
    hcclTp2NativeExchange  = 1
} hcclTp2NativeMode;

typedef struct
{
    uint32_t longSoIndex;
    uint64_t targetValue;
} hcclTp2NativeSyncInfo;

typedef struct
{
    hcclTp2NativeGraphState state;
    uint64_t                preparedReplayCount;
    uint64_t                nativeReplayCount;
    uint64_t                capturedCommandCount;
    uint64_t                capturedStreamCount;
    uint64_t                capturedRelocationCount;
    uint64_t                dependencyRelocationCount;
    uint64_t                wqeRelocationCount;
    uint64_t                externalCompletionAddressRelocationCount;
    uint64_t                internalCompletionAddressRelocationCount;
    uint64_t                capturedByteCount;
    uint64_t                maxStreamBytes;
    uint64_t                streamCcbBytes;
    uint64_t                nativeReplayBytes;
    uint64_t                nativeCcbWrapCount;
    uint64_t                nativeSubmissionCount;
    uint64_t                wrapCapturedCommandCount;
    uint64_t                completionRingSize;
    uint64_t                nativeWrapReplayCount;
    hcclTp2NativeSyncInfo   completion;
} hcclTp2NativeGraphInfo;

hcclResult_t hcclTp2NativeGraphCreate(const void*               sendbuff,
                                      void*                     recvbuff,
                                      size_t                    count,
                                      hcclDataType_t            datatype,
                                      hcclRedOp_t               reduceOp,
                                      hcclComm_t                comm,
                                      void*                     streamHandle,
                                      hcclTp2NativeMode         mode,
                                      hcclTp2NativeGraphHandle* graphHandle);
hcclResult_t hcclTp2NativeGraphCapture(hcclTp2NativeGraphHandle graphHandle);

// Prepared replay is an intermediate qualification path: it reuses the
// instantiated primitive graph and skips the public HCCL dispatcher, but still
// regenerates scheduler commands. It is deliberately separate from the final
// packet-template replay API so it cannot be reported as native replay.
hcclResult_t hcclTp2NativeGraphReplayPrepared(hcclTp2NativeGraphHandle graphHandle,
                                             const hcclTp2NativeSyncInfo* dependency,
                                             hcclTp2NativeSyncInfo*       completion);
// Per-collective native path: patches dependency/WQE fields and publishes the
// microstream templates without HcclGraph::submit(). It is not whole-chain replay.
hcclResult_t hcclTp2NativeGraphReplay(hcclTp2NativeGraphHandle   graphHandle,
                                     const hcclTp2NativeSyncInfo* dependency,
                                     hcclTp2NativeSyncInfo*       completion);
hcclResult_t hcclTp2NativeGraphStage(hcclTp2NativeGraphHandle    graphHandle,
                                    const hcclTp2NativeSyncInfo* dependency,
                                    hcclTp2NativeSyncInfo*       completion);
hcclResult_t hcclTp2NativeGraphSubmit(hcclTp2NativeGraphHandle graphHandle);
hcclResult_t hcclTp2NativeGraphGetInfo(hcclTp2NativeGraphHandle graphHandle,
                                      hcclTp2NativeGraphInfo*   graphInfo);
hcclResult_t hcclTp2NativeGraphDestroy(hcclTp2NativeGraphHandle graphHandle);

// One prepared program per microstream and completion-ring phase. Handles are
// exclusively leased until batch destruction; no per-node Stage API is used.
hcclResult_t hcclTp2NativeBatchCreate(const hcclTp2NativeGraphHandle* graphs, size_t count,
                                     hcclTp2NativeBatchHandle* batch);
hcclResult_t hcclTp2NativeBatchReplay(hcclTp2NativeBatchHandle batch,
                                     const hcclTp2NativeSyncInfo* dependencies, size_t count,
                                     hcclTp2NativeSyncInfo* completions);
// Cold-configure one fixed publication boundary. Segmented replay then accepts
// exactly prefix followed by suffix; the ordinary whole-batch path remains valid.
hcclResult_t hcclTp2NativeBatchConfigureSplit(hcclTp2NativeBatchHandle batch,
                                             size_t prefixCount);
hcclResult_t hcclTp2NativeBatchReplaySegmented(hcclTp2NativeBatchHandle batch,
                                              const hcclTp2NativeSyncInfo* dependencies,
                                              size_t count,
                                              hcclTp2NativeSyncInfo* completions);
hcclResult_t hcclTp2NativeBatchDestroy(hcclTp2NativeBatchHandle batch);


#ifdef __cplusplus
}
#endif
