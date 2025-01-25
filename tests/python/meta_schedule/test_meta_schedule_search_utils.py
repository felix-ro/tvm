# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for search utilities"""
from typing import List

from tvm import meta_schedule as ms
from tvm.tir import Schedule
from tvm.target import Target
from tvm.meta_schedule.testing.space_generation import generate_design_space
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.meta_schedule.tune_context import _normalize_mod

from tvm.meta_schedule.search_strategy.search_utils import sample_initial_population

def _target():
    return Target("llvm --num-cores=16")

def _design_space(mod):
    return generate_design_space(
        kind="llvm",
        mod=mod,
        target=_target(),
        types=ms.ScheduleRule,
    )

def test_sample_initial_population():
    """Test for initial population sampling"""
    mod = _normalize_mod(create_te_workload("C1D", 0))
    design_spaces: List[Schedule] = _design_space(mod)

    design_space_traces = []
    for space in design_spaces:
        design_space_traces.append(space.trace)

    population: List[Schedule] = sample_initial_population([], design_space_traces, mod, 5)
    print(len(population))

    for sample in population:
        trace = sample.trace
        if trace is not None:
            trace.show()

if __name__ == "__main__":
    test_sample_initial_population()
