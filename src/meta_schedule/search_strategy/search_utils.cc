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

using tvm::runtime::Array;

Array<tir::Schedule> SampleInitialPopulation(Array<Postproc>& postprocs,
                                             Array<tir::Trace>& design_spaces,
                                             IRModule mod,
                                             PackedFunc logger,
                                             int population_size,
                                             int num_threads) { 
    ThreadedTraceApply pp(postprocs);
    std::vector<tir::Schedule> population(population_size);

    auto SampleSchedules = [&](int thread_id, int task_id) -> void {
        TRandState rand_state = support::LinearCongruentialEngine::DeviceRandom();
        int design_space_index = tir::SampleInt(&rand_state, 0, design_spaces.size());
        tir::Trace trace(design_spaces[design_space_index]->insts, {});

        tir::Schedule& result = population.at(task_id);
        if (Optional<tir::Schedule> sch = pp.Apply(mod, trace, &rand_state)) {
            result = sch.value();
        }
    };
    support::parallel_for_dynamic(0, population_size, num_threads, SampleSchedules);
    TVM_PY_LOG(INFO, logger) << "SampleInitialPopulation summary:\n"
                                         << pp.SummarizeFailures();
    return Array<tir::Schedule>(population);
}

TVM_REGISTER_GLOBAL("meta_schedule.SampleInitialPopulation")
    .set_body_typed([](Array<Postproc> postprocs, Array<tir::Trace> design_spaces, IRModule mod,
                       PackedFunc logger, int population_size, int num_threads) {
      return SampleInitialPopulation(postprocs, design_spaces, mod, logger, population_size, num_threads);
    });
} // meta_schedule
} // tvm