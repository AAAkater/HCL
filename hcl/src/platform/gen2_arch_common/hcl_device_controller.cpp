#include "infra/scal/gen2_arch_common/cyclic_buffer_manager.h"
#include "platform/gen2_arch_common/hcl_device_controller.h"
#include "infra/scal/gen2_arch_common/scal_manager.h"         // for Gen2Arc...
#include "platform/gen2_arch_common/commands/hcl_commands.h"  // for HclComm...
#include "platform/gen2_arch_common/hcl_device.h"             // for HclDevi...
#include "platform/gen2_arch_common/hcl_packets_utils.h"      // for SoBaseAndSize, getCompCfg
#include "infra/scal/gen2_arch_common/scal_names.h"
#include "infra/scal/gen2_arch_common/scal_utils.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>

bool HclDeviceControllerGen2Arch::NativeGraphCapture::validate() const
{
    if (streams.empty() || externalTargetDelta == 0 || capturedExternalCompletionAddress == 0 ||
        relocationCount(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS) == 0)
    {
        return false;
    }
    if (internalTargetDelta != 0 &&
        (capturedInternalCompletionAddress == 0 ||
         relocationCount(hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS) == 0))
    {
        return false;
    }
    for (const NativeStreamCapture& stream : streams)
    {
        if (stream.commandTemplate.empty())
        {
            return false;
        }
        if (!hcl::validateNativeHclStreamTemplate(stream.commandTemplate))
        {
            return false;
        }
    }
    return true;
}

HclDeviceControllerGen2Arch::HclDeviceControllerGen2Arch(const unsigned numOfStreams) : m_numOfStreams(numOfStreams)
{
    m_graphSync        = std::make_unique<std::unique_ptr<HclGraphSyncGen2Arch>[]>(m_numOfStreams);
    m_streamSyncParams = new ArchStreamSyncParams[m_numOfStreams];

    for (unsigned i = 0; i < m_numOfStreams; i++)
    {
        m_streamSyncParams[i].m_longtermGPSOManager   = new CreditManager(GCFG_HCL_LONGTERM_GPSO_COUNT.value());
        m_streamSyncParams[i].m_requestedExtraCredits = 0;
    }
}

HclDeviceControllerGen2Arch::~HclDeviceControllerGen2Arch()
{
    for (unsigned i = 0; i < m_numOfStreams; i++)
    {
        if (m_streamSyncParams[i].m_regularGPSOManager != nullptr) delete m_streamSyncParams[i].m_regularGPSOManager;
        if (m_streamSyncParams[i].m_longtermGPSOManager != nullptr) delete m_streamSyncParams[i].m_longtermGPSOManager;
        if (m_streamSyncParams[i].m_hfcMonitorManager != nullptr) delete m_streamSyncParams[i].m_hfcMonitorManager;
    }

    delete[] m_streamSyncParams;
}

HclCommandsGen2Arch& HclDeviceControllerGen2Arch::getGen2ArchCommands()
{
    return *m_commands;
}

hcl::Gen2ArchScalManager& HclDeviceControllerGen2Arch::getGen2ArchScalManager()
{
    return *m_scalManager;
}

void HclDeviceControllerGen2Arch::setupCompCfg(int archStreamId)
{
    LOG_HCL_TRACE(HCL, "Stream({})", archStreamId);
    uint32_t soBase = 0xffffffff;
    size_t   soSize = 0;
    for (unsigned gpIdx = 0; gpIdx < (unsigned)GpsoPool::COUNT; gpIdx++)
    {
        soBase = soBase > m_graphSync[archStreamId]->getSoPoolBaseAddr(gpIdx)
                     ? m_graphSync[archStreamId]->getSoPoolBaseAddr(gpIdx)
                     : soBase;
        soSize += m_graphSync[archStreamId]->getSoPoolSize(gpIdx);
    }
    getCompCfg()[archStreamId * 2] =
        SoBaseAndSize(m_scalManager->getCgInfo(archStreamId)[(int)hcl::SchedulerType::external].cgBaseAddr,
                      m_scalManager->getCgInfo(archStreamId)[(int)hcl::SchedulerType::external].size);
    getCompCfg()[archStreamId * 2 + 1] = SoBaseAndSize(soBase, soSize);
}

void HclDeviceControllerGen2Arch::initDeviceForCollectiveRoutine(int            archStreamId,
                                                                 hcl::syncInfo* longSo,
                                                                 hcl::syncInfo* longSoNullSubmit)
{
    LOG_HCL_TRACE(HCL,
                  "Stream({}), LongSO({}, {}, 0x{:x})",
                  archStreamId,
                  longSo->long_so_index,
                  longSo->targetValue,
                  (uint64_t)(longSo->cp_handle));
    auto& syncParams              = getSyncParams(archStreamId);
    syncParams.m_longSo           = longSo;
    syncParams.m_longSoNullSubmit = longSoNullSubmit;
    auto& smInfo                  = syncParams.m_smInfo;
    auto& deviceSimbPoolManager   = m_device->getDeviceSimbPoolManager(archStreamId);
    m_graphSync[archStreamId]->setSyncData(smInfo.soBaseIdx, smInfo.soSize);

    std::array<hcl::CgInfo, (int)hcl::SchedulerType::count> cgInfo = m_scalManager->getCgInfo(archStreamId);
    m_graphSync[archStreamId]->setCgInfo(cgInfo[(int)hcl::SchedulerType::external],
                                         cgInfo[(int)hcl::SchedulerType::internal],
                                         GCFG_HCL_LONGTERM_GPSO_COUNT.value(),
                                         deviceSimbPoolManager.getPoolBufferSize(SCALEUP_AND_ALL2ALL_POOL));
    longSo->long_so_index = cgInfo[(int)hcl::SchedulerType::external].longSoIndex;
    longSo->targetValue   = cgInfo[(int)hcl::SchedulerType::external].longSoInitialValue;
    longSo->cp_handle     = m_scalManager->getCgHandle(archStreamId, true);

    LOG_HCL_TRACE(HCL,
                  "Initialized LongSO({}, {}, 0x{:x})",
                  longSo->long_so_index,
                  longSo->targetValue,
                  (uint64_t)(longSo->cp_handle));

    *longSoNullSubmit = *longSo;

    syncParams.m_regularGPSOManager = new CreditManager(m_graphSync[archStreamId]->getSoPoolSize(GpsoPool::GPSO_0));

    setupHFCMonitors(archStreamId, syncParams);

    for (unsigned poolContainerIndex = 0; poolContainerIndex < deviceSimbPoolManager.getPoolContainerCount();
         poolContainerIndex++)
    {
        m_scalManager->addContainerParamsPerStream(
            deviceSimbPoolManager.getPoolContainerParamsPerStream(poolContainerIndex));
    }

    const unsigned monitorsPerSched     = smInfo.monitorSize / SCHED_COUNT;
    const unsigned longMonitorsPerSched = ((smInfo.longMonitorSize / 4) / SCHED_COUNT) * 4;

    for (unsigned i = 0; i < SCHED_COUNT; ++i)
    {
        syncParams.m_schedulers[i].internalResources.monitorBase  = smInfo.monitorBaseIdx + (i * monitorsPerSched);
        syncParams.m_schedulers[i].internalResources.monitorsSize = monitorsPerSched;
        syncParams.m_schedulers[i].internalResources.longMonitorBase =
            smInfo.longMonitorBaseIdx + (i * longMonitorsPerSched);
        syncParams.m_schedulers[i].internalResources.longMonitorsSize = longMonitorsPerSched;
    }

    // To ensure external and internal Cgs are synced, we allocate all the SOs in the external Cg
    allocAllExternalBarrier(archStreamId);

    // setup all monitors once
    setupMonitors(archStreamId);

    setupCompCfg(archStreamId);
}

void HclDeviceControllerGen2Arch::setupHFCMonitors(int archStreamId, ArchStreamSyncParams& syncParams)
{
    // doesn't really matter the scalStream, as we don't have a dedicated stream for init
    hcl::ScalStream&      scalStream = getScalStream(archStreamId, HclStreamIndex::SO_SEND_ARB);
    HclGraphSyncGen2Arch& graphSync  = getGraphSync(archStreamId);
    const int64_t         cgSize     = graphSync.getCgData(false).size;
    syncParams.m_hfcMonitorManager   = new CreditManager(cgSize);
    unsigned int        numMonitors  = syncParams.m_hfcMonitorManager->getPoolSize();
    unsigned int        smIdx        = syncParams.m_smInfo.monitorSmIndex;
    unsigned int        monitorBase  = syncParams.m_smInfo.hfcMonitorBaseIdx;
    hcl::HostFenceInfo& fenceInfo    = getGen2ArchScalManager().getHostFenceInfo(archStreamId, 0).hostFenceInfo;
    uint64_t fenceAddr = graphSync.getFullRegSobObj(graphSync.getSyncManagerBase(fenceInfo.smDcore), fenceInfo.smIndex);
    graphSync.addSetupHFCMonitors(scalStream, monitorBase, numMonitors, graphSync.getSyncManagerBase(smIdx), fenceAddr);
}

void HclDeviceControllerGen2Arch::allocAllExternalBarrier(int archStreamId)
{
    hcl::CgInfo      externalCgInfo = m_graphSync[archStreamId]->getCgData(true);
    hcl::ScalStream& currentStream  = getScalStream(archStreamId, HclStreamIndex::GC);
    currentStream.setTargetValue(m_streamSyncParams[archStreamId].m_longSo->targetValue);
    m_commands->serializeAllocBarrierCommand(currentStream,
                                             currentStream.getSchedIdx(),
                                             externalCgInfo.cgIdx[currentStream.getSchedIdx()],
                                             externalCgInfo.size);
}

/**
 * @brief Setup monitors for all micro architecture streams.
 *
 * This function initializes the monitors for each micro architecture stream.
 * It ensures that there are enough monitors and long monitors available for each stream.
 * It then sets up the monitors and long monitors for each stream.
 *
 * @param archStreamId The architecture stream ID.
 */
void HclDeviceControllerGen2Arch::setupMonitors(int archStreamId)
{
    for (unsigned hclStreamIndex = 0; hclStreamIndex < m_streamLayout->getTotalMicroArchStreamCount(); hclStreamIndex++)
    {
        hcl::ScalStream& scalStream    = getScalStream(archStreamId, (HclStreamIndex)hclStreamIndex);
        unsigned         schedIdx      = scalStream.getSchedIdx();
        unsigned         uarchStreamId = scalStream.getUarchStreamIndex();

        auto& schedResources = m_streamSyncParams[archStreamId].m_schedulers[schedIdx].internalResources;

        // Ensure there are enough monitors
        VERIFY(schedResources.monitorsSize >= MONITORS_PER_STREAM * MAX_STREAM_PER_SCHED,
               "There aren't enough monitors");

        // Ensure there are enough long monitors
        VERIFY(schedResources.longMonitorsSize >= LONG_MONITOR_LENGTH * LONG_MONITORS_PER_STREAM * MAX_STREAM_PER_SCHED,
               "There aren't enough long monitors");

        unsigned fenceBase = getFenceIdx(archStreamId, uarchStreamId, FENCE_MONITOR_IDX);
        scalStream.setTargetValue(m_streamSyncParams[archStreamId].m_longSo->targetValue);

        // Setup regular monitors
        for (unsigned fenceIdx = 0; fenceIdx < FENCES_PER_STREAM; ++fenceIdx)
        {
            uint64_t monitorPayloadAddr =
                m_scalManager->getMonitorPayloadAddr((hcl::SchedulersIndex)schedIdx, fenceBase + fenceIdx);

            m_graphSync[archStreamId]->addSetupMonitors(scalStream,
                                                        uarchStreamId,
                                                        schedResources.monitorBase,
                                                        m_streamSyncParams[archStreamId].m_smInfo.monitorSmIndex,
                                                        monitorPayloadAddr,
                                                        fenceBase,
                                                        fenceIdx);
        }

        // Setup long monitors
        for (unsigned fenceIdx = 0; fenceIdx < LONG_MONITORS_PER_STREAM; ++fenceIdx)
        {
            uint64_t monitorPayloadAddr =
                m_scalManager->getMonitorPayloadAddr((hcl::SchedulersIndex)schedIdx, fenceBase + fenceIdx);

            m_graphSync[archStreamId]->addSetupLongMonitors(
                scalStream,
                m_streamSyncParams[archStreamId].m_smInfo.longMonitorSmIndex,
                monitorPayloadAddr,
                getLongMonitorIdx(archStreamId, schedIdx, uarchStreamId),
                fenceBase,
                fenceIdx);
        }
    }
}

void HclDeviceControllerGen2Arch::advanceProg(int archStreamId, bool nopOp)
{
    m_streamSyncParams[archStreamId].m_longSo->targetValue++;

    if (!GCFG_HCL_NULL_SUBMIT.value() || nopOp)
    {
        m_graphSync[archStreamId]->incSoIndex(1);
    }

    m_device->getDeviceSimbPoolManager(archStreamId)
        .advanceProg(m_streamSyncParams[archStreamId].m_longSo->targetValue);

    m_streamSyncParams[archStreamId].m_longtermGPSOManager->advanceProg(
        m_streamSyncParams[archStreamId].m_longSo->targetValue);

    // Indicates that we had an api call after event wait, so in the next event record Nop is not needed.
    m_streamSyncParams[archStreamId].m_isPrevWaitEvent = false;
}

unsigned int HclDeviceControllerGen2Arch::handleExtraCredits(int archStreamId, unsigned extraCreditsNeeded)
{
    unsigned creditsToAsk = 1;
    auto&    syncParams   = getSyncParams(archStreamId);
    if (extraCreditsNeeded >= syncParams.m_requestedExtraCredits)
    {
        creditsToAsk += extraCreditsNeeded - syncParams.m_requestedExtraCredits;
        syncParams.m_requestedExtraCredits += (extraCreditsNeeded - syncParams.m_requestedExtraCredits);
    }
    else if (syncParams.m_requestedExtraCredits)
    {
        --syncParams.m_requestedExtraCredits;
        creditsToAsk = 0;
    }

    LOG_HCL_TRACE(HCL, "extraCreditsNeeded={}, creditsToAsk={}", extraCreditsNeeded, creditsToAsk);
    return creditsToAsk;
}

void HclDeviceControllerGen2Arch::nopArbStreamRecipe(int                  archStreamId,
                                                     hcl::SchedulersIndex schedIdx,
                                                     unsigned             requiredCredits,
                                                     uint64_t             targetValue,
                                                     hcl::ScalStream&     garbageCollectorStream)
{
    // After waiting for the required credits, the arbitrator on the gc scheduler needs to un-fence the gc stream
    llvm_vecsmall::SmallVector<unsigned, MAX_STREAM_PER_SCHED> streamsToInc = {};
    if (garbageCollectorStream.getSchedIdx() == (unsigned)schedIdx)
        streamsToInc.push_back(garbageCollectorStream.getUarchStreamIndex());

    // Wait for credits on the arb stream of each scheduler
    auto& arbStream = getScalArbUarchStream(archStreamId, schedIdx);
    arbStream.setTargetValue(targetValue);
    setOpExecutionConditions(arbStream, requiredCredits, streamsToInc);

    // Set cmax on the external SO
    if (schedIdx == hcl::SchedulersIndex::sendScaleUp)
    {
        int additionalSignalExternal = m_graphSync[archStreamId]->isForceOrder(true)
                                           ? m_device->getSignalsCalculator().signalToCost(SignalEvent::FORCE_ORDER)
                                           : 0;

        m_commands->serializeLbwWriteCommand(
            arbStream,
            arbStream.getSchedIdx(),
            m_graphSync[archStreamId]->getCurrentCgSoAddr(CgType::eExternal),
            m_graphSync[archStreamId]->getSoConfigValue(m_scalManager->getCMaxTargetValue() - additionalSignalExternal,
                                                        true));
    }
}

void HclDeviceControllerGen2Arch::addNop(int archStreamId)
{
    advanceProg(archStreamId, true);
    auto& syncParams = getSyncParams(archStreamId);
    syncParams.m_longSoNullSubmit->targetValue++;

    uint64_t targetValue =
        GCFG_HCL_NULL_SUBMIT.value() ? syncParams.m_longSoNullSubmit->targetValue : syncParams.m_longSo->targetValue;

    LOG_HCL_CONTEXT_INFO(HCL, "Running Nop command, targetValue={}", syncParams.m_longSo->targetValue);

    LOG_TRACE(HCL_CG,
              SCAL_PROGRESS_HCL_FMT "addNop",
              archStreamId,
              syncParams.m_longSo->long_so_index,
              syncParams.m_longSo->targetValue);

    const unsigned requiredCredits = GCFG_HCL_NULL_SUBMIT.value() ? 1 : handleExtraCredits(archStreamId, 0);

    // the long SO and the GPSO pool are tightly coupled so we need to move the gpso idx
    syncParams.m_regularGPSOManager->allocNextCredit(syncParams.m_longSo->targetValue);

    auto& garbageCollectorStream = getScalStream(archStreamId, HclStreamIndex::GC);
    for (unsigned i = 0; i < hcl::SchedulersIndex::count; i++)
    {
        nopArbStreamRecipe(archStreamId, (hcl::SchedulersIndex)i, requiredCredits, targetValue, garbageCollectorStream);
    }

    ////////////////////////////////////////////////// GC STREAM RECIPE ///////////////////////////////////////////
    garbageCollectorStream.setTargetValue(targetValue);
    waitForExecutionConditions(garbageCollectorStream);

    // scheduler waits for credits on the external CG
    setGcExecutionConditions(garbageCollectorStream, 1, {});

    // set cmax on the internal SO
    int additionalSignalInternal = m_graphSync[archStreamId]->isForceOrder(false)
                                       ? m_device->getSignalsCalculator().signalToCost(SignalEvent::FORCE_ORDER)
                                       : 0;
    m_commands->serializeLbwWriteCommand(
        garbageCollectorStream,
        garbageCollectorStream.getSchedIdx(),
        m_graphSync[archStreamId]->getCurrentCgSoAddr(CgType::eInternal),
        m_graphSync[archStreamId]->getSoConfigValue(m_scalManager->getCMaxTargetValue() - additionalSignalInternal,
                                                    true));
    incInternalCgTargetValue(archStreamId);
}

void HclDeviceControllerGen2Arch::submitWork(int archStreamId, bool submitToHw)
{
    auto& syncParams = getSyncParams(archStreamId);

    if (submitToHw)
    {
        for (unsigned hclStreamIndex = 0; hclStreamIndex < m_streamLayout->getTotalMicroArchStreamCount();
             hclStreamIndex++)
        {
            hcl::ScalStream& scalStream = getScalStream(archStreamId, (HclStreamIndex)hclStreamIndex);
            if (scalStream.requiresSubmission())
            {
                scalStream.submit();
            }
        }

        uint64_t targetValue = GCFG_HCL_NULL_SUBMIT.value() ? syncParams.m_longSoNullSubmit->targetValue
                                                            : syncParams.m_longSo->targetValue;
        if (targetValue - syncParams.m_submittedTargetValue)
        {
            // External CG
            scal_completion_group_set_expected_ctr(m_scalManager->getCgHandle(archStreamId, true), targetValue);
        }
        syncParams.m_submittedTargetValue = syncParams.m_longSo->targetValue;

        if (syncParams.m_InternalCgTargetValue - syncParams.m_submittedInternalCgTargetValue)
        {
            // Internal CG
            scal_completion_group_set_expected_ctr(m_scalManager->getCgHandle(archStreamId, false),
                                                   syncParams.m_InternalCgTargetValue);
        }
        syncParams.m_submittedInternalCgTargetValue = syncParams.m_InternalCgTargetValue;
    }

    syncParams.m_signalFinalize();
}

void HclDeviceControllerGen2Arch::beginNativeCapture(int archStreamId)
{
    auto& syncParams = getSyncParams(archStreamId);
    VERIFY(!syncParams.m_nativeCaptureActive, "Native HCL graph capture is already active");
    VERIFY(syncParams.m_longSo != nullptr, "Native HCL graph requires an initialized completion group");
    syncParams.m_nativeCaptureActive        = true;
    syncParams.m_nativeCaptureExternalStart = syncParams.m_longSo->targetValue;
    syncParams.m_nativeCaptureInternalStart = syncParams.m_InternalCgTargetValue;
    for (unsigned streamIndex = 0; streamIndex < m_streamLayout->getTotalMicroArchStreamCount(); ++streamIndex)
    {
        getScalStream(archStreamId, static_cast<HclStreamIndex>(streamIndex)).beginNativeCapture();
    }
}

HclDeviceControllerGen2Arch::NativeGraphCapture HclDeviceControllerGen2Arch::endNativeCapture(int archStreamId)
{
    auto& syncParams = getSyncParams(archStreamId);
    VERIFY(syncParams.m_nativeCaptureActive, "Native HCL graph capture is not active");
    NativeGraphCapture capture;
    for (unsigned streamIndex = 0; streamIndex < m_streamLayout->getTotalMicroArchStreamCount(); ++streamIndex)
    {
        hcl::NativeHclStreamTemplate commandTemplate =
            getScalStream(archStreamId, static_cast<HclStreamIndex>(streamIndex)).endNativeCapture();
        if (!commandTemplate.empty())
        {
            capture.streams.push_back(
                {.streamIndex = static_cast<HclStreamIndex>(streamIndex),
                 .commandTemplate = std::move(commandTemplate)});
        }
    }
    capture.externalTargetDelta = syncParams.m_longSo->targetValue - syncParams.m_nativeCaptureExternalStart;
    capture.internalTargetDelta =
        syncParams.m_InternalCgTargetValue - syncParams.m_nativeCaptureInternalStart;
    capture.completionLongSoIndex = syncParams.m_longSo->long_so_index;
    capture.capturedExternalCompletionAddress =
        m_graphSync[archStreamId]->getCurrentCgSoAddr(CgType::eExternal);
    capture.capturedInternalCompletionAddress =
        m_graphSync[archStreamId]->getCurrentCgSoAddr(CgType::eInternal);
    capture.capturedGpso0Address =
        m_graphSync[archStreamId]->getCurrentGpsoAddr(WaitMethod::GPSO_0);
    capture.capturedGpso1Address =
        m_graphSync[archStreamId]->getCurrentGpsoAddr(WaitMethod::GPSO_1);
    size_t externalAddressRelocations = 0;
    size_t internalAddressRelocations = 0;
    for (NativeStreamCapture& streamCapture : capture.streams)
    {
        for (auto& relocation : streamCapture.commandTemplate.relocations)
        {
            if (relocation.kind == hcl::NativeHclRelocationKind::PENDING_COMPLETION_ADDRESS ||
                relocation.kind == hcl::NativeHclRelocationKind::PENDING_COMPLETION_INDEX ||
                relocation.kind == hcl::NativeHclRelocationKind::PENDING_SHORT_MONITOR)
            {
                const bool indexed = relocation.kind == hcl::NativeHclRelocationKind::PENDING_COMPLETION_INDEX;
                const bool shortMonitor = relocation.kind == hcl::NativeHclRelocationKind::PENDING_SHORT_MONITOR;
                const uint32_t addresses[] = {capture.capturedExternalCompletionAddress,
                    capture.capturedInternalCompletionAddress, capture.capturedGpso0Address, capture.capturedGpso1Address};
                const hcl::NativeHclRelocationKind addressKinds[] = {
                    hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS,
                    hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS,
                    hcl::NativeHclRelocationKind::GPSO_0_ADDRESS, hcl::NativeHclRelocationKind::GPSO_1_ADDRESS};
                const hcl::NativeHclRelocationKind indexKinds[] = {
                    hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_INDEX,
                    hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_INDEX,
                    hcl::NativeHclRelocationKind::GPSO_0_INDEX, hcl::NativeHclRelocationKind::GPSO_1_INDEX};
                const hcl::NativeHclRelocationKind monitorKinds[] = {
                    hcl::NativeHclRelocationKind::EXTERNAL_SHORT_MONITOR,
                    hcl::NativeHclRelocationKind::INTERNAL_SHORT_MONITOR,
                    hcl::NativeHclRelocationKind::GPSO_0_SHORT_MONITOR,
                    hcl::NativeHclRelocationKind::GPSO_1_SHORT_MONITOR};
                size_t pool = 0;
                while (pool < 4 && relocation.capturedAddress != addresses[pool]) ++pool;
                VERIFY(pool < 4, "Untracked native completion or short-monitor address 0x{:x}", relocation.capturedAddress);
                relocation.kind = shortMonitor ? monitorKinds[pool] : indexed ? indexKinds[pool] : addressKinds[pool];
                if (indexed) capture.capturedEdmaBases[pool] = getSoIdxBaseIdx(addresses[pool]).baseIdx;
                if (shortMonitor)
                {
                    const auto& command = streamCapture.commandTemplate.commands.at(relocation.commandIndex);
                    uint32_t encoded = 0;
                    VERIFY(relocation.byteOffset + sizeof(encoded) <= command.bytes.size(),
                           "Native short-monitor relocation lies outside its command");
                    std::memcpy(&encoded, command.bytes.data() + relocation.byteOffset, sizeof(encoded));
                    const uint32_t index = hcl::nativeHclShortMonitorIndex(encoded);
                    auto& captured = capture.capturedShortMonitorIndexes[pool];
                    VERIFY(captured == UINT32_MAX || captured == index, "Native short-monitor pool index changed");
                    captured = index;
                }
                continue;
            }
            if (relocation.kind != hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS ||
                relocation.mask != 0x07ffffffu)
                continue;
            const auto& command = streamCapture.commandTemplate.commands.at(relocation.commandIndex);
            uint32_t encoded = 0;
            VERIFY(relocation.byteOffset + sizeof(encoded) <= command.bytes.size(),
                   "Native NIC SOB relocation lies outside its command");
            std::memcpy(&encoded, command.bytes.data() + relocation.byteOffset, sizeof(encoded));
            // RS signals its internal reduction GPSO before the external
            // completion. Classify the same tracked SOB pools used by LBW
            // relocations; never reinterpret an unknown NIC destination.
            const std::pair<uint32_t, hcl::NativeHclRelocationKind> pools[] = {
                {capture.capturedExternalCompletionAddress, hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS},
                {capture.capturedInternalCompletionAddress, hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS},
                {capture.capturedGpso0Address, hcl::NativeHclRelocationKind::GPSO_0_ADDRESS},
                {capture.capturedGpso1Address, hcl::NativeHclRelocationKind::GPSO_1_ADDRESS}};
            bool found = false;
            for (const auto& pool : pools)
            {
                if ((encoded & relocation.mask) != (pool.first & relocation.mask)) continue;
                VERIFY(!found, "Native NIC SOB matches multiple completion pools");
                relocation.kind = pool.second;
                found = true;
            }
            VERIFY(found, "Untracked native NIC SOB 0x{:x}: external=0x{:x} internal=0x{:x} gpso0=0x{:x} gpso1=0x{:x}",
                   encoded, capture.capturedExternalCompletionAddress, capture.capturedInternalCompletionAddress,
                   capture.capturedGpso0Address, capture.capturedGpso1Address);
        }
        externalAddressRelocations += hcl::addNativeHclCapturedWordRelocations(
            streamCapture.commandTemplate,
            capture.capturedExternalCompletionAddress,
            hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS);
        if (capture.internalTargetDelta != 0)
        {
            internalAddressRelocations += hcl::addNativeHclCapturedWordRelocations(
                streamCapture.commandTemplate,
                capture.capturedInternalCompletionAddress,
                hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS);
        }
        hcl::addNativeHclCapturedWordRelocations(
            streamCapture.commandTemplate,
            capture.capturedGpso0Address,
            hcl::NativeHclRelocationKind::GPSO_0_ADDRESS);
        hcl::addNativeHclCapturedWordRelocations(
            streamCapture.commandTemplate,
            capture.capturedGpso1Address,
            hcl::NativeHclRelocationKind::GPSO_1_ADDRESS);
        streamCapture.program = hcl::NativeHclProgram::prepare(
            {{&streamCapture.commandTemplate, 0}},
            getScalStream(archStreamId, streamCapture.streamIndex).nativeCommandAlignment());
    }
    syncParams.m_nativeCaptureActive = false;
    // Count relocations emitted by the packet serializers too. An AllGather
    // may signal only a packed NIC/EDMA SOB rather than an extra full LBW word.
    externalAddressRelocations = capture.relocationCount(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS);
    internalAddressRelocations = capture.relocationCount(hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS);
    VERIFY(externalAddressRelocations != 0,
           "Native HCL graph did not capture its external completion SOB address 0x{:x}",
           capture.capturedExternalCompletionAddress);
    VERIFY(capture.internalTargetDelta == 0 || internalAddressRelocations != 0,
           "Native HCL graph did not capture its internal completion SOB address 0x{:x}",
           capture.capturedInternalCompletionAddress);
    VERIFY(capture.validate(), "Captured native HCL graph has an invalid command or completion contract");
    capture.executableValidated = true;
    return capture;
}

void HclDeviceControllerGen2Arch::cancelNativeCapture(int archStreamId)
{
    for (unsigned streamIndex = 0; streamIndex < m_streamLayout->getTotalMicroArchStreamCount(); ++streamIndex)
    {
        getScalStream(archStreamId, static_cast<HclStreamIndex>(streamIndex)).cancelNativeCapture();
    }
    getSyncParams(archStreamId).m_nativeCaptureActive = false;
}

hcl::NativeHclReplayState HclDeviceControllerGen2Arch::prepareNativeReplayState(
    int                              archStreamId,
    const NativeGraphCapture&        capture,
    const hcl::NativeHclReplayState& replayState)
{
    VERIFY(capture.executableValidated, "Cannot replay a native HCL graph before instantiation");
    auto& syncParams = getSyncParams(archStreamId);
    VERIFY(!syncParams.m_nativeCaptureActive, "Cannot replay native HCL commands during capture");
    VERIFY(syncParams.m_longSo != nullptr, "Native HCL graph requires an initialized completion group");
    VERIFY(syncParams.m_longSo->long_so_index == capture.completionLongSoIndex,
           "Native HCL graph completion group changed from {} to {}",
           capture.completionLongSoIndex,
           syncParams.m_longSo->long_so_index);

    const uint64_t nextExternalTarget = syncParams.m_longSo->targetValue + capture.externalTargetDelta;
    const uint64_t nextInternalTarget = syncParams.m_InternalCgTargetValue + capture.internalTargetDelta;

    hcl::NativeHclReplayState patchedState = replayState;
    patchedState.values[static_cast<size_t>(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_TARGET)] =
        nextExternalTarget;
    patchedState.values[static_cast<size_t>(hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_TARGET)] =
        nextInternalTarget;
    m_graphSync[archStreamId]->incSoIndex(static_cast<unsigned>(capture.externalTargetDelta));
    patchedState.values[static_cast<size_t>(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS)] =
        m_graphSync[archStreamId]->getCurrentCgSoAddr(CgType::eExternal);
    patchedState.values[static_cast<size_t>(hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS)] =
        m_graphSync[archStreamId]->getCurrentCgSoAddr(CgType::eInternal);
    patchedState.values[static_cast<size_t>(hcl::NativeHclRelocationKind::GPSO_0_ADDRESS)] =
        m_graphSync[archStreamId]->getCurrentGpsoAddr(WaitMethod::GPSO_0);
    patchedState.values[static_cast<size_t>(hcl::NativeHclRelocationKind::GPSO_1_ADDRESS)] =
        m_graphSync[archStreamId]->getCurrentGpsoAddr(WaitMethod::GPSO_1);
    const hcl::NativeHclRelocationKind addressKinds[] = {
        hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS,
        hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS,
        hcl::NativeHclRelocationKind::GPSO_0_ADDRESS, hcl::NativeHclRelocationKind::GPSO_1_ADDRESS};
    const hcl::NativeHclRelocationKind indexKinds[] = {
        hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_INDEX,
        hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_INDEX,
        hcl::NativeHclRelocationKind::GPSO_0_INDEX, hcl::NativeHclRelocationKind::GPSO_1_INDEX};
    const hcl::NativeHclRelocationKind monitorKinds[] = {
        hcl::NativeHclRelocationKind::EXTERNAL_SHORT_MONITOR,
        hcl::NativeHclRelocationKind::INTERNAL_SHORT_MONITOR,
        hcl::NativeHclRelocationKind::GPSO_0_SHORT_MONITOR,
        hcl::NativeHclRelocationKind::GPSO_1_SHORT_MONITOR};
    const uint32_t capturedAddresses[] = {
        capture.capturedExternalCompletionAddress, capture.capturedInternalCompletionAddress,
        capture.capturedGpso0Address, capture.capturedGpso1Address};
    for (size_t pool = 0; pool < 4; ++pool)
    {
        if (capture.capturedShortMonitorIndexes[pool] != UINT32_MAX)
            patchedState.values[static_cast<size_t>(monitorKinds[pool])] = hcl::nativeHclShortMonitorBits(
                capture.capturedShortMonitorIndexes[pool], capturedAddresses[pool],
                static_cast<uint32_t>(patchedState.get(addressKinds[pool])));
        if (capture.capturedEdmaBases[pool] == UINT32_MAX) continue;
        const auto location = getSoIdxBaseIdx(static_cast<uint32_t>(patchedState.get(addressKinds[pool])));
        VERIFY(location.baseIdx == capture.capturedEdmaBases[pool] && location.soIdx <= 0x3ff,
               "Native EDMA completion moved outside its captured base/index range");
        patchedState.values[static_cast<size_t>(indexKinds[pool])] = location.soIdx;
    }

    const char* diagnostics = std::getenv("HCL_TP2_NATIVE_REPLAY_DIAGNOSTICS");
    if (diagnostics != nullptr && std::strcmp(diagnostics, "1") == 0)
    {
        std::fprintf(stderr,
                     "NATIVE_HCL_REPLAY dependency=%llu:%llu external_target=%llu->%llu "
                     "internal_target=%llu->%llu external_address=0x%llx->0x%llx "
                     "internal_address=0x%llx->0x%llx gpso0=0x%llx->0x%llx gpso1=0x%llx->0x%llx "
                     "relocations=%zu/%zu/%zu/%zu/%zu\n",
                     static_cast<unsigned long long>(
                         replayState.get(hcl::NativeHclRelocationKind::DEPENDENCY_LONG_SO_INDEX)),
                     static_cast<unsigned long long>(
                         replayState.get(hcl::NativeHclRelocationKind::DEPENDENCY_TARGET)),
                     static_cast<unsigned long long>(syncParams.m_longSo->targetValue),
                     static_cast<unsigned long long>(nextExternalTarget),
                     static_cast<unsigned long long>(syncParams.m_InternalCgTargetValue),
                     static_cast<unsigned long long>(nextInternalTarget),
                     static_cast<unsigned long long>(capture.capturedExternalCompletionAddress),
                     static_cast<unsigned long long>(
                         patchedState.get(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS)),
                     static_cast<unsigned long long>(capture.capturedInternalCompletionAddress),
                     static_cast<unsigned long long>(
                         patchedState.get(hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS)),
                     static_cast<unsigned long long>(capture.capturedGpso0Address),
                     static_cast<unsigned long long>(patchedState.get(hcl::NativeHclRelocationKind::GPSO_0_ADDRESS)),
                     static_cast<unsigned long long>(capture.capturedGpso1Address),
                     static_cast<unsigned long long>(patchedState.get(hcl::NativeHclRelocationKind::GPSO_1_ADDRESS)),
                     capture.relocationCount(hcl::NativeHclRelocationKind::DEPENDENCY_TARGET),
                     capture.relocationCount(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS),
                     capture.relocationCount(hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS),
                     capture.relocationCount(hcl::NativeHclRelocationKind::GPSO_0_ADDRESS),
                     capture.relocationCount(hcl::NativeHclRelocationKind::GPSO_1_ADDRESS));
        std::fflush(stderr);
        LOG_INFO(HCL,
                 "Native HCL replay dependency {}:{} external target {}->{} internal target {}->{} "
                 "external completion address 0x{:x}->0x{:x} internal completion address 0x{:x}->0x{:x} "
                 "relocations dependency={} external-address={} internal-address={}",
                 replayState.get(hcl::NativeHclRelocationKind::DEPENDENCY_LONG_SO_INDEX),
                 replayState.get(hcl::NativeHclRelocationKind::DEPENDENCY_TARGET),
                 syncParams.m_longSo->targetValue,
                 nextExternalTarget,
                 syncParams.m_InternalCgTargetValue,
                 nextInternalTarget,
                 capture.capturedExternalCompletionAddress,
                 patchedState.get(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS),
                 capture.capturedInternalCompletionAddress,
                 patchedState.get(hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS),
                 capture.relocationCount(hcl::NativeHclRelocationKind::DEPENDENCY_TARGET),
                 capture.relocationCount(hcl::NativeHclRelocationKind::EXTERNAL_COMPLETION_ADDRESS),
                 capture.relocationCount(hcl::NativeHclRelocationKind::INTERNAL_COMPLETION_ADDRESS));
    }

    syncParams.m_longSo->targetValue = nextExternalTarget;
    syncParams.m_InternalCgTargetValue = nextInternalTarget;
    return patchedState;
}

hcl::syncInfo HclDeviceControllerGen2Arch::replayNativeCapture(
    int archStreamId, const NativeGraphCapture& capture,
    const hcl::NativeHclReplayState& replayState, bool submit)
{
    const auto state = prepareNativeReplayState(archStreamId, capture, replayState);
    for (const auto& streamCapture : capture.streams)
        getScalStream(archStreamId, streamCapture.streamIndex).replayNativeProgram(streamCapture.program, &state, 1);
    if (submit) submitNativeReplay(archStreamId);
    return *getSyncParams(archStreamId).m_longSo;
}

HclDeviceControllerGen2Arch::NativeBatchProgram HclDeviceControllerGen2Arch::prepareNativeBatch(
    int archStreamId, const std::vector<const NativeGraphCapture*>& captures)
{
    VERIFY(!captures.empty() && captures.size() <= 256, "Invalid native NIC batch size");
    std::map<HclStreamIndex, std::vector<hcl::NativeHclProgram::Input>> byStream;
    for (size_t node = 0; node < captures.size(); ++node)
    {
        VERIFY(captures[node] && captures[node]->executableValidated, "Native NIC batch input is unprepared");
        for (const auto& stream : captures[node]->streams)
            byStream[stream.streamIndex].push_back({&stream.commandTemplate, static_cast<uint32_t>(node)});
    }
    NativeBatchProgram result;
    for (const auto& entry : byStream)
    {
        const auto alignment = getScalStream(archStreamId, entry.first).nativeCommandAlignment();
        // The bounded batch is published in full before compute starts. Every
        // producer/consumer wait remains in the command bytes; captured host
        // PI writes are unnecessary here. Single-graph replay keeps them.
        auto program = hcl::NativeHclProgram::prepare(entry.second, alignment, false);
        // Keep one batch below half the CCB, including worst alignment padding.
        // Two divisions cover the existing PI/CI lag workaround. Therefore any
        // CCB recycle wait while writing this batch belongs to earlier,
        // already-published work, never to its as-yet-unpublished compute peer.
        const auto ccb = hcl::ScalStream::getCcbSize();
        const auto division = ccb / hcl::CyclicBufferManager::m_numberOfDivisions;
        VERIFY(program.pages().size() + 1 <= (ccb / 2 - 2 * division) / alignment,
               "Native NIC batch exceeds bounded publication capacity");
        result.push_back({entry.first, std::move(program)});
    }
    return result;
}

void HclDeviceControllerGen2Arch::publishNativeBatch(int archStreamId,
    const NativeBatchProgram& program, const hcl::NativeHclReplayState* states, size_t stateCount)
{
    for (const auto& stream : program)
        getScalStream(archStreamId, stream.streamIndex).replayNativeProgram(stream.program, states, stateCount);
    submitNativeReplay(archStreamId);
}

void HclDeviceControllerGen2Arch::submitNativeReplay(int archStreamId)
{
    auto& syncParams = getSyncParams(archStreamId);
    VERIFY(syncParams.m_longSo != nullptr, "Native HCL graph requires an initialized completion group");
    for (unsigned streamIndex = 0; streamIndex < m_streamLayout->getTotalMicroArchStreamCount(); ++streamIndex)
    {
        hcl::ScalStream& stream = getScalStream(archStreamId, static_cast<HclStreamIndex>(streamIndex));
        if (stream.requiresSubmission())
        {
            stream.submit();
        }
    }

    scal_completion_group_set_expected_ctr(m_scalManager->getCgHandle(archStreamId, true),
                                           syncParams.m_longSo->targetValue);
    syncParams.m_submittedTargetValue = syncParams.m_longSo->targetValue;
    if (syncParams.m_InternalCgTargetValue != syncParams.m_submittedInternalCgTargetValue)
    {
        scal_completion_group_set_expected_ctr(m_scalManager->getCgHandle(archStreamId, false),
                                               syncParams.m_InternalCgTargetValue);
        syncParams.m_submittedInternalCgTargetValue = syncParams.m_InternalCgTargetValue;
    }
    syncParams.m_signalFinalize();
}

HclDeviceControllerGen2Arch::NativeReplayCounters HclDeviceControllerGen2Arch::nativeReplayCounters(
    int                       archStreamId,
    const NativeGraphCapture& capture)
{
    NativeReplayCounters counters;
    for (const NativeStreamCapture& streamCapture : capture.streams)
    {
        const hcl::ScalStream& stream = getScalStream(archStreamId, streamCapture.streamIndex);
        const uint64_t         pi     = stream.getUnwrappedPi();
        counters.bytes += pi;
        counters.ccbWraps += pi / hcl::ScalStream::getCcbSize();
        counters.submissions += stream.getSubmissionCount();
    }
    return counters;
}

void HclDeviceControllerGen2Arch::streamAddWait(hcl::ScalStream&            scalStream,
                                                const SyncObjectDescriptor& descriptor,
                                                bool                        useEqual)
{
    unsigned       archStreamIdx = scalStream.getArchStreamIndex();
    const unsigned waitingMonitorBase =
        m_streamSyncParams[archStreamIdx].m_schedulers[scalStream.getSchedIdx()].internalResources.monitorBase;

    m_graphSync[archStreamIdx]->createSyncStreamsMessages(
        scalStream,
        waitingMonitorBase,
        m_streamSyncParams[archStreamIdx].m_smInfo.soSmIndex,
        descriptor.value,
        descriptor.sob.sobId,
        getFenceIdx(archStreamIdx, scalStream.getUarchStreamIndex(), FENCE_MONITOR_IDX),
        useEqual);
}

void HclDeviceControllerGen2Arch::setExecutionConditions(
    hcl::ScalStream&                                                  scalStream,
    bool                                                              external,
    unsigned                                                          creditsNr,
    const llvm_vecsmall::SmallVector<unsigned, MAX_STREAM_PER_SCHED>& streamsToInc,
    bool                                                              shouldAddWait,
    LBWBurstData_t*                                                   lbwBurstData)

{
    unsigned                                                   archStreamIdx = scalStream.getArchStreamIndex();
    unsigned                                                   schedIdx      = scalStream.getSchedIdx();
    llvm_vecsmall::SmallVector<unsigned, MAX_STREAM_PER_SCHED> fences;

    if (shouldAddWait)
    {
        addStreamWaitOnExternalEvent(scalStream, ARB_STREAM_IDX);
    }

    if (creditsNr)
    {
        const hcl::CgInfo& cgInfo = m_graphSync[archStreamIdx]->getCgData(external);
        for (unsigned i = 0; i < streamsToInc.size(); i++)
        {
            fences.push_back(getFenceIdx(archStreamIdx, streamsToInc[i], FENCE_BARRIER_IDX));
        }

        /* the AllocBarrierCommand waits for credits on the required CG.
        in order to deal with scheduler command limits, we also allowed to append fence increments and LBWs to it.
        the fence Incs are usually related and free streams that used waitForExecutionAuthorization.
        the LBWs are usually related to the CG and are used to signal the completion of the CG.
        */
        m_commands->serializeAllocBarrierCommand(scalStream,
                                                 schedIdx,
                                                 cgInfo.cgIdx[schedIdx],
                                                 creditsNr,
                                                 &fences,
                                                 lbwBurstData);
    }
    else
    {
        for (unsigned i = 0; i < streamsToInc.size(); i++)
        {
            m_commands->serializeFenceIncCommand(scalStream,
                                                 schedIdx,
                                                 getFenceIdx(archStreamIdx, streamsToInc[i], FENCE_BARRIER_IDX));
        }
        if (lbwBurstData)
        {
            m_commands->serializeLbwWriteCommand(scalStream,
                                                 schedIdx,
                                                 (*lbwBurstData)[0].addr,
                                                 (*lbwBurstData)[0].data);
        }
    }
}

void HclDeviceControllerGen2Arch::waitForExecutionConditions(hcl::ScalStream& scalStream)
{
    m_commands->serializeFenceDecCommand(
        scalStream,
        scalStream.getSchedIdx(),
        getFenceIdx(scalStream.getArchStreamIndex(), scalStream.getUarchStreamIndex(), FENCE_BARRIER_IDX));
}

void HclDeviceControllerGen2Arch::setHostFences(int     archStreamId,
                                                int     uarchStreamId,
                                                bool    isSend,
                                                uint8_t scaleoutInternalFences,
                                                llvm_vecsmall::SmallVector<FenceInfo, HOST_FENCES_NR>& scaleoutFences)
{
    /*
        ---------------------------------------------------
        |      | RS uarchStreamIdx=0 | AG uarchStreamIdx=1 |
        ----------------------------------------------
        | Send |    Fence=0,4,8      | Fence=2,6,10        |
        ----------------------------------------------
        | Recv |    Fence=1,5,9      | Fence=3,7,11        |
        ---------------------------------------------------
     */
    VERIFY(scaleoutInternalFences <= 1, "Can only provide 1 Host fence per send/recv slice");
    for (uint8_t i = 0; i < scaleoutInternalFences; i++)
    {
        const unsigned fenceIndex = (isSend ? 0 : 1) + 2 * uarchStreamId;
        VERIFY(fenceIndex < HOST_FENCES_NR, "Illegal host fence index");
        hcl::HostFenceInfo& fenceInfo = m_scalManager->getHostFenceInfo(archStreamId, fenceIndex).hostFenceInfo;
        LOG_HCL_TRACE(HCL,
                      "archStream={}, uarchstreamIdx={}, fenceIndex={}, smDcore={}, smIndex={},",
                      archStreamId,
                      uarchStreamId,
                      fenceIndex,
                      fenceInfo.smDcore,
                      fenceInfo.smIndex);
        scaleoutFences.push_back(
            {fenceIndex,
             {m_graphSync[archStreamId]->getRegSobObj(m_graphSync[archStreamId]->getSyncManagerBase(fenceInfo.smDcore),
                                                      fenceInfo.smIndex),
              m_graphSync[archStreamId]->getSoConfigValue(1, true)}});
    }
}

hcl::syncInfo HclDeviceControllerGen2Arch::eventRecord(int      archStreamId,
                                                       bool     isCollectTime /*= false*/,
                                                       uint64_t timestampHandle /*= 0*/,
                                                       uint32_t timestampsOffset /*= 0*/)
{
    auto&                       syncParams = getSyncParams(archStreamId);
    std::lock_guard<std::mutex> lock(syncParams.m_streamLock);
    if (syncParams.m_isPrevWaitEvent || GCFG_HCL_NULL_SUBMIT.value())
    {
        addNop(archStreamId);
        submitWork(archStreamId);
    }
    LOG_TRACE(HCL_CG,
              SCAL_PROGRESS_HCL_FMT "eventRecord",
              archStreamId,
              syncParams.m_longSo->long_so_index,
              syncParams.m_longSo->targetValue);

    hcl::syncInfo longSo = GCFG_HCL_NULL_SUBMIT.value() ? *syncParams.m_longSoNullSubmit : *syncParams.m_longSo;

    if (isCollectTime)
    {
        m_scalManager->cgRegisterTimeStemp(archStreamId, longSo.targetValue, timestampHandle, timestampsOffset);
    }

    syncParams.m_isPrevWaitEvent = false;

    return longSo;
}

void HclDeviceControllerGen2Arch::streamWaitEvent(int archStreamId, hcl::syncInfo commonState)
{
    std::lock_guard<std::mutex> lock(m_streamSyncParams[archStreamId].m_streamLock);

    LOG_TRACE(HCL_CG,
              SCAL_PROGRESS_HCL_FMT "streamWaitEvent",
              archStreamId,
              commonState.long_so_index,
              commonState.targetValue);

    m_streamSyncParams[archStreamId].m_isPrevWaitEvent = true;
    m_graphSync[archStreamId]->addPendingWait(commonState.long_so_index, commonState.targetValue);
}

void HclDeviceControllerGen2Arch::synchronizeStream(int archStreamId)
{
    hcl::syncInfo longSo = eventRecord(archStreamId);
    m_scalManager->synchronizeStream(archStreamId, longSo.targetValue);
}

bool HclDeviceControllerGen2Arch::streamQuery(int archStreamId)
{
    hcl::syncInfo longSo = eventRecord(archStreamId);
    return m_scalManager->streamQuery(archStreamId, longSo.targetValue);
}

void HclDeviceControllerGen2Arch::enableNullSubmit(int archStreamId, bool enable)
{
    m_scalManager->disableCcb(archStreamId, enable);
    m_graphSync[archStreamId]->setNullSubmit(enable);
}

void HclDeviceControllerGen2Arch::setTraceMarker(int archStreamId, uint32_t val)
{
    LOG_HCL_TRACE(HCL, "setTraceMarker = {}", val);
    hcl::ScalStream& currentStream = getScalStream(archStreamId, HclStreamIndex::SU_SEND_ARB);
    addStreamWaitOnExternalEvent(currentStream, currentStream.getUarchStreamIndex());

    m_commands->serializeSetTraceMarker(currentStream, currentStream.getSchedIdx(), val);
    submitWork(archStreamId);
}

void HclDeviceControllerGen2Arch::setTraceMarker(int archStreamId, HclStreamIndex streamIdx, uint32_t val)
{
    LOG_HCL_TRACE(HCL, "setTraceMarker:{}, HclStreamIndex:{}", val, streamIdx);
    hcl::ScalStream& currentStream = getScalStream(archStreamId, streamIdx);

    m_commands->serializeSetTraceMarker(currentStream, currentStream.getSchedIdx(), val);
}
