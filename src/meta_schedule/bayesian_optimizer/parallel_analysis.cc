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
#include <algorithm>
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

using tir::InstructionNode;
using tir::Instruction;
using tir::Trace;
using tir::Schedule;

/*! \brief The candidate to be mutated */
struct Candidate {
  /*! \brief The annotation instruction */
  Instruction inst;
  /*! \brief The current parallel extent */
  int64_t parallel_extent;
  /*! \brief The name of the root block */
  String block_name;
  /*! \brief The name of the PrimFunc */
  String func_name;
};

/*!
 * \brief Get an instruction that annotates the maximum parallel extent
 * \param trace The trace to be mutated
 * \param rand_state The random state
 * \param candidate The candidate to be mutated
 * \param ann_inst 
 * \return Whether a decision is found
 */
bool FindParallelDecision(const Trace& trace, TRandState* rand_state,
                          Candidate* candidate, const InstructionNode* ann_inst) {
  using tir::BlockRVNode;
  using tir::InstructionNode;
  
  std::unordered_map<const BlockRVNode*, const InstructionNode*> get_block_insts;
  get_block_insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (const BlockRVNode* block_rv = tir::GetInstGetBlockOutput(inst)) {
      get_block_insts[block_rv] = inst.get();
    }
  }

  ICHECK_EQ(ann_inst->inputs.size(), 2);
  const InstructionNode* get_block_inst =
      get_block_insts.at(Downcast<tir::BlockRV>(ann_inst->inputs[0]).get());
  ICHECK_EQ(get_block_inst->attrs.size(), 2);

  candidate->inst = GetRef<Instruction>(ann_inst);
  candidate->parallel_extent = Downcast<IntImm>(ann_inst->inputs[1])->value;
  candidate->block_name = Downcast<String>(get_block_inst->attrs[0]);
  candidate->func_name = Downcast<String>(get_block_inst->attrs[1]);
  return true;
}

/*!
 * \brief Get all possible parallel annotate decisions
 * \param sch The schedule the annotation is in
 * \param trace The trace with the parallel annotate instruction
 * \param rand_state The random state
 * \param ann_inst The parallel annotate instruction
 * \param max_parallel_extent The maximum parallel extent (num_cores * jobs_per_core)
 * \return All possible annotation values
 */
Array<Integer> GetPossibleParallelAnnotateDecisions(const Schedule& sch, const Trace& trace,
                                     TRandState* rand_state, Instruction* ann_inst,
                                     const int64_t max_parallel_extent_) {
  // Step 1. Find a parallel decision.
  Candidate candidate;
  const InstructionNode* inst = ann_inst->get();
  FindParallelDecision(trace, rand_state, &candidate, inst);

  // Step 2. Find all possible parallel plans.
  std::vector<std::vector<int64_t>> loop_extent_prods = tir::AnalyzeParallel(
      sch->state(), candidate.block_name, candidate.func_name, max_parallel_extent_);
  std::unordered_map<int64_t, std::vector<int>> limit2plan;
  std::map<std::vector<int>, int64_t> plan2limit;
  for (const std::vector<int64_t>& prods : loop_extent_prods) {
    for (int64_t limit : prods) {
      if (limit <= max_parallel_extent_ && !limit2plan.count(limit)) {
        std::vector<int> plan = tir::GetNumFusedLoops(loop_extent_prods, limit);
        limit2plan[limit] = plan;
        plan2limit[plan] = limit;
      }
    }
  }
  
  // Step 3. Prepare array with all possible decisions.
  Array<Integer> arr;
  for (const auto& pair : plan2limit) {
    arr.push_back(pair.second);
  }

  return arr; 
}

TVM_REGISTER_GLOBAL("tir.analysis.get_possible_parallel_annotate_decisions").set_body_typed([](
  const Schedule sch, const Trace trace, TRandState rand_state,
  Instruction ann_inst, const Integer max_parallel_extent) {

  return GetPossibleParallelAnnotateDecisions(sch, trace, &rand_state, &ann_inst, max_parallel_extent.IntValue());
});              

/*!
 * \brief Changes the annotation value of an instruction in a trace
 * \param trace The trace which includes the annotate instruction
 * \param ann_inst The annotate instruction
 * \param ann_val The new annotation value
 * \return The updated Trace
 */
Trace ChangeAnnotationInTrace(const Trace trace, const Instruction ann_inst, const int64_t ann_val) {
  Array<Instruction> insts;
  insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (inst.same_as(ann_inst)) {
      insts.push_back(tir::ReplaceAnnValue(ann_inst, ann_val));
    } else if (inst->kind->IsPostproc()) {
      break;
    } else {
      insts.push_back(inst);
    }
  }
  return Trace(insts, trace->decisions);
}

TVM_REGISTER_GLOBAL("tir.schedule.change_annotation_in_trace").set_body_typed([](
  const Trace trace, Instruction ann_inst, const Integer ann_val) {
  return ChangeAnnotationInTrace(trace, ann_inst, ann_val.IntValue());
});     

}  // namespace meta_schedule
}  // namespace tvm
