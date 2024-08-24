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
"""Utility functions to enable quick search strategy prototyping"""
from typing import TYPE_CHECKING, List, Optional
from tvm.tir.schedule import Schedule, Trace
from tvm import IRModule

from .. import _ffi_api
from ..utils import cpu_count
from ..logging import Logger, get_logging_func, get_logger

if TYPE_CHECKING:
    from .. postproc import Postproc


def sample_initial_population(postprocs: List["Postproc"],
                              design_spaces: List[Trace],
                              mod: IRModule,
                              population_size: int,
                              logger: Optional[Logger] = None,
                              num_threads: Optional[int] = None) -> List[Schedule]:
    """ Samples a population of schedules with random decisions from the design spaces

    Parameters
    ----------
    postprocs: List[Postproc]
        The post processors to use to validate a schedules correctness
    design_spaces: List[Trace]
        The design spaces to sample from
    mod: IRModule
        The IRModule on which we apply the transformation trace
    populations_size: int
        The number of schedules to sample
    logger: Optional[Logger]
        The logger which records the postprocessing statistics
    num_threads: Optional[int]
        The number of threads to use to create the schedules. Set to logical CPU count by default.

    Returns
    -------
    sampled_population: List[Schedules]
        The sampled schedules
    """
    if num_threads is None:
        num_threads = cpu_count()

    if logger is None:
        logger = get_logger(__name__)

    return _ffi_api.SampleInitialPopulation(postprocs, design_spaces, mod, # type: ignore # pylint: disable=no-member
                                            get_logging_func(logger), population_size, num_threads)
        