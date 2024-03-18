/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

using tir::Schedule;
using tir::Instruction;
using tir::BlockRV;

std::vector<int> CollectComputeLocationIndices(const Schedule sch, const BlockRV block) {
    tir::ScheduleState sch_state = sch->state();
    tir::StmtSRef block_sref = sch->GetSRef(Downcast<tir::BlockRV>(block));
    auto [location_srefs, location_indices] = tir::CollectComputeLocation(sch_state, block_sref);

    return location_indices;
}

TVM_REGISTER_GLOBAL("tir.analysis.collect_compute_location_indices").set_body_typed([](
  const Schedule sch, const BlockRV block) {
  std::vector<int> locations = CollectComputeLocationIndices(sch, block);

  // turn vector into array for packed function return
  // (there must be a better way to handle this)
  Array<Integer> arr;
  for (const int location: locations) {
    arr.push_back(location);
  }

  return arr;
});    

} // namespace meta_schedule
} // namespace tvm