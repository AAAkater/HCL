#include "collective_interface/collectives/all_reduce.h"

#include "collective_interface/prims/hccl_prim.h"
#include "collective_interface/hccl_graph.h"
#include "collective_interface/prims/simple_prims.h"
#include "collective_interface/prims/scaleup_prims.h"
#include "hcl_global_conf.h"
#include "hccl/tp2_direct_api_context.h"

static hcclResult_t ar_runTp2ReplicatedReduceScatter(IHcclGraphEngine* engine,
                                                     HclCollectiveParams& params,
                                                     bool directNicTransfer,
                                                     bool directNicReduction)
{
    params.m_currentOp = eHCLReduceScatter;
    params.m_directScaleupNicTransfer = directNicTransfer;
    params.m_directScaleupNicReduction = directNicReduction;
    HcclGraph graph(engine, &params);
    graph.createPrim<HcclPrimReduceScatter>(ReduceScatterPrimArgs {
        {params.m_sendBufferAddr, params.m_recvBufferAddr, {}, params.m_count},
        false,
        true,
        directNicTransfer,
        directNicReduction,
    });
    return graph.submit();
}

std::unique_ptr<HcclGraph> ar_createTp2DirectGraph(IHcclGraphEngine* engine,
                                                  HclCollectiveParams& params,
                                                  bool directNicReduction)
{
    params.m_currentOp = eHCLReduceScatter;
    params.m_directScaleupNicTransfer = true;
    params.m_directScaleupNicReduction = directNicReduction;
    auto graph = std::make_unique<HcclGraph>(engine, &params);
    graph->createPrim<HcclPrimReduceScatter>(ReduceScatterPrimArgs {
        {params.m_sendBufferAddr, params.m_recvBufferAddr, {}, params.m_count},
        false,
        true,
        true,
        directNicReduction,
    });
    graph->instantiate();
    return graph;
}

hcclResult_t ar_runPairwise(IHcclGraphEngine* engine, HclCollectiveParams& params)
{
    const HCL_Rank myRank           = params.m_dynamicComm.getMyRank();
    const uint16_t myBox            = params.m_dynamicComm.getMyScaleupGroup();
    const uint32_t commSize         = params.m_dynamicComm.getCommSize();
    const uint32_t scaleupGroupSize = params.m_dynamicComm.getScaleupGroupSize();
    const uint32_t boxesCount       = commSize / scaleupGroupSize;
    const uint64_t countPerRank     = params.m_count / commSize;
    const uint64_t rankAddrOffset   = countPerRank * dataTypeSizeInBytes(params.m_dataType);

    const bool dedicatedDirectRequested = HclGen2::isTp2DedicatedDirectCall();
    const bool dedicatedExchangeRequested = HclGen2::isTp2DedicatedExchangeCall();
    const bool directNicExchangeRequested =
        dedicatedExchangeRequested ||
        (!dedicatedDirectRequested && GCFG_HCL_TP2_DIRECT_NIC_EXCHANGE.value());
    const bool directNicRequested =
        dedicatedDirectRequested || GCFG_HCL_TP2_DIRECT_NIC_RS_AR.value() || directNicExchangeRequested;
    const bool inPlace = params.m_sendBufferAddr == params.m_recvBufferAddr;
    const bool useInPlaceStaging = directNicRequested &&
                                   GCFG_HCL_TP2_DIRECT_INPLACE_STAGING.value() && inPlace;
    const bool useDirectOutOfPlace = directNicRequested &&
                                     !GCFG_HCL_TP2_DIRECT_INPLACE_STAGING.value() && !inPlace;
    const bool useDirectNicExchange = useDirectOutOfPlace && directNicExchangeRequested;
    const bool useDirectNicReduction = useInPlaceStaging ||
                                       (useDirectOutOfPlace && !useDirectNicExchange);
    const bool useDirectNicTransfer = useDirectNicReduction || useDirectNicExchange;
    const bool validBufferLayout = useDirectNicTransfer || inPlace;
    if ((GCFG_HCL_TP2_REPLICATED_RS_AR.value() || useDirectNicTransfer) && commSize == 2 &&
        scaleupGroupSize == 2 && boxesCount == 1 && validBufferLayout && params.m_dataType == hcclBfloat16 &&
        params.m_reduceOp == hcclSum && params.m_count > 0)
    {
        return ar_runTp2ReplicatedReduceScatter(
            engine, params, useDirectNicTransfer, useDirectNicReduction);
    }

    {
        params.m_currentOp = eHCLReduceScatter;
        HcclGraph graph(engine, &params);

        BufferToken scaleoutBuff = graph.generateBufferToken(STATIC_BUFFER);

        if (boxesCount > 1)
        {
            bool castUp = isDataTypeTwoBytes(params.m_dataType);
            graph.createPrim<HcclPrimReduceScatter>(
                ReduceScatterPrimArgs {{params.m_sendBufferAddr + myBox * scaleupGroupSize * rankAddrOffset,
                                        0,
                                        scaleoutBuff,
                                        countPerRank * scaleupGroupSize},
                                       castUp});
        }
        else
        {
            graph.createPrim<HcclPrimReduceScatter>(ReduceScatterPrimArgs {
                {params.m_sendBufferAddr + myBox * scaleupGroupSize * rankAddrOffset,
                 params.m_recvBufferAddr + params.m_dynamicComm.getRankInScaleupGroup() * rankAddrOffset,
                 {},
                 countPerRank * scaleupGroupSize},
                false});
        }

        for (uint32_t i = 1; i < boxesCount; i++)
        {
            HCL_Rank sendRank = (myRank + i * scaleupGroupSize) % commSize;
            HCL_Rank recvRank = (myRank - i * scaleupGroupSize + commSize) % commSize;

            uint16_t sendBox     = params.m_dynamicComm.getRankToScaleupGroupMap()[sendRank];
            uint64_t boxOffset   = sendBox * scaleupGroupSize * rankAddrOffset;
            uint64_t rsInputAddr = params.m_sendBufferAddr + boxOffset;

            BufferToken scaleupBuff = graph.generateBufferToken(TEMP_BUFFER);

            auto rs = graph.createPrim<HcclPrimReduceScatter>(
                ReduceScatterPrimArgs {{rsInputAddr, 0, scaleupBuff, countPerRank * scaleupGroupSize}});

            const bool doReduction = true;

            auto send =
                graph.createPrim<HcclPrimSend>(SendPrimArgs {sendRank, 0, scaleupBuff, countPerRank, doReduction});

            graph.addWait(rs, send);

            bool cast = isDataTypeTwoBytes(params.m_dataType);
            auto recv = graph.createPrim<HcclPrimRecv>(
                RecvPrimArgs {recvRank, 0, scaleoutBuff, countPerRank, doReduction, cast});

            if (i == boxesCount - 1)
            {
                uint64_t reductionDstAddr = params.m_recvBufferAddr + myBox * scaleupGroupSize * rankAddrOffset +
                                            params.m_dynamicComm.getRankInScaleupGroup() * rankAddrOffset;
                auto reduction = graph.createPrim<HcclPrimReduction>(
                    ReductionPrimArgs {0, scaleoutBuff, reductionDstAddr, countPerRank, cast});
                graph.addWait(recv, reduction);
            }
        }
        graph.submit();
    }

    {
        params.m_currentOp = eHCLAllGather;
        HcclGraph graph(engine, &params);
        graph.startStrongOrder = true;
        graph.createPrim<HcclPrimAllGather>(params.m_recvBufferAddr + myBox * scaleupGroupSize * rankAddrOffset,
                                            params.m_recvBufferAddr + myBox * scaleupGroupSize * rankAddrOffset,
                                            countPerRank);

        for (uint32_t i = 1; i < boxesCount; i++)
        {
            HCL_Rank sendRank = (myRank + i * scaleupGroupSize) % commSize;
            HCL_Rank recvRank = (myRank - i * scaleupGroupSize + commSize) % commSize;

            uint64_t recvAddr    = params.m_recvBufferAddr + recvRank * rankAddrOffset;
            uint16_t recvBox     = params.m_dynamicComm.getRankToScaleupGroupMap()[recvRank];
            uint64_t boxOffset   = recvBox * scaleupGroupSize * rankAddrOffset;
            uint64_t agInOutAddr = params.m_recvBufferAddr + boxOffset;

            auto recv = graph.createPrim<HcclPrimRecv>(RecvPrimArgs {recvRank, recvAddr, {}, countPerRank});
            auto ag   = graph.createPrim<HcclPrimAllGather>(agInOutAddr, agInOutAddr, countPerRank);
            graph.addWait(recv, ag);

            graph.createPrim<HcclPrimSend>(
                SendPrimArgs {sendRank, params.m_recvBufferAddr + myRank * rankAddrOffset, {}, countPerRank});
        }
        return graph.submit();
    }
}

// Retain the ordinary four-rank AllReduce arithmetic: ReduceScatter followed
// by in-place AllGather. Each graph owns one HCL completion, so it can reuse
// the existing native capture/relocation and completion-ring machinery.
std::unique_ptr<HcclGraph> ar_createTp4NativePhaseGraph(IHcclGraphEngine* engine,
                                                      HclCollectiveParams& params, int phase)
{
    const auto& comm = params.m_dynamicComm;
    if (comm.getCommSize() != 4 || comm.getScaleupGroupSize() != 4 || phase < 0 || phase > 2 ||
        params.m_count == 0 || (phase != 2 && params.m_count % 4 != 0)) return nullptr;
    params.m_directScaleupNicTransfer = false;
    params.m_directScaleupNicReduction = false;
    params.m_currentOp = phase == 0 ? eHCLReduceScatter : eHCLAllGather;
    params.m_collectiveOp = phase == 2 ? eHCLAllGather : eHCLAllReduce;
    if (phase == 2) params.m_reduceOp = hcclOpNone;
    auto graph = std::make_unique<HcclGraph>(engine, &params);
    if (phase == 0)
    {
        const uint64_t perRank = params.m_count / 4;
        const uint64_t offset = comm.getRankInScaleupGroup() * perRank * dataTypeSizeInBytes(params.m_dataType);
        graph->createPrim<HcclPrimReduceScatter>(ReduceScatterPrimArgs {
            {params.m_sendBufferAddr, params.m_recvBufferAddr + offset, {}, params.m_count}, false});
    }
    else
    {
        graph->startStrongOrder = phase == 1;
        graph->createPrim<HcclPrimAllGather>(params.m_sendBufferAddr, params.m_recvBufferAddr,
                                            phase == 1 ? params.m_count / 4 : params.m_count);
    }
    graph->instantiate();
    return graph;
}
