#pragma once
#include "hccl_types.h"
#include "hcl_collective_params.h"

#include <memory>

class IHcclGraphEngine;
class HcclGraph;

hcclResult_t ar_runPairwise(IHcclGraphEngine* engine, HclCollectiveParams& params);
std::unique_ptr<HcclGraph> ar_createTp2DirectGraph(IHcclGraphEngine* engine,
                                                  HclCollectiveParams& params,
                                                  bool directNicReduction);

// One existing stock phase per native completion; phases: RS, AR-AG, standalone AG.
std::unique_ptr<HcclGraph> ar_createTp4NativePhaseGraph(IHcclGraphEngine* engine,
                                                      HclCollectiveParams& params, int phase);
