from typing import TYPE_CHECKING, List, Optional, Any, Dict, Union, Tuple
from tvm.tir.schedule import Schedule, Trace, Instruction, BlockRV
from tvm.tir.analysis import (is_annotate_with_parallel,
                              get_possible_parallel_annotate_decisions,
                              collect_compute_location_indices)
from tvm.ir import IRModule, make_node
from tvm.runtime import String
from tvm.tir import IntImm

from .search_strategy import PySearchStrategy, MeasureCandidate, SearchStrategy
from ..utils import derived_object, cpu_count
from ..arg_info import ArgInfo
from ..runner import RunnerResult
from ..logging import get_logger, get_logging_func
from ..profiler import Profiler

from ..cost_model import CostModel

if TYPE_CHECKING:
    from ..database import Database, TuningRecord
    from ..tune_context import TuneContext
    from ..postproc import Postproc

import numpy as np
import random
import functools
import operator
import logging
import inspect
from itertools import permutations
import hashlib
import os
import shutil
import json
import time
import re

from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs, NotUniqueError

DECISION_TYPE = Any
ATTR_TYPE = Any

decision_lookup = dict()
logger = get_logging_func(get_logger(__name__))


def create_hash(input_string: str) -> str:
    """Returns the hash-str of an input string

    Parameters
    ----------
    input_string: str
        The input string

    Returns
    -------
    hash_string: str
        The corresponding hash-str
    """
    input_bytes = input_string.encode('utf-8')
    hash_obj = hashlib.sha256(input_bytes)
    return hash_obj.hexdigest()


def current_line_number() -> int:
    """Returns the line number this function is called on

    Returns
    -------
    line_number: int
        The line number the function is called on
    """
    return inspect.currentframe().f_back.f_lineno


def forkseed(rand_state: int) -> int:
    """Creates a new random state from an initial seed

    Parameters
    ----------
    rand_state: int
        The initial random state

    Returns
    -------
    new_rand_state: int
        The new random state
    """
    rand_state = int(rand_state+random.random()*1999999973)
    return (rand_state * 32767) % 1999999973


# ToDo rework this function
def sample_int(rand_state: np.int64, min_inclusive: int, max_exclusive: int):
    assert min_inclusive < max_exclusive, "ValueError: max_exclusive must be greater than min_inclusive."

    if ((min_inclusive + 1) == max_exclusive):
        return min_inclusive
    rand_ = forkseed(rand_state)
    # call np.random to generate [min, max-1]
    np.random.seed(rand_)
    dist = random.randint(min_inclusive, max_exclusive-1)
    return dist


def get_top_k_schedules(context: "TuneContext", cost_model: CostModel,
                        schedules: List[Schedule], k: int) -> Union[List[Schedule] | List[float]]:
    """Returns the top-k Schedules in a list of Schedules based on a cost model

    Parameters
    ----------
    context: tvm.meta_schedule.TuneContext
        The tuning context
    cost_model: tvm.meta_schedule.CostModel
        The cost model to use for the selection
    schedules: List[tvm.schedule.Schedule]
        The list of Schedules to select from
    k: int
        The number of Schedules to select

    Returns
    -------
    top_schedules: List[tvm.schedule.Schedule]
        The top-k Schedules in the list of Schedules
    top_scores: List[float]
        The corresponding scores to the schedules
    """
    with Profiler.timeit("BayOptSearch/GetTopKSchedules"):
        scores = predict_normalized_scores(schedules, context, cost_model)
        idx = np.argsort(scores)[-k:][::-1]

        top_schedules: List[Schedule] = []
        top_scores: List[float] = []
        for index in idx:
            top_schedules.append(schedules[index])
            top_scores.append(scores[index])
        return top_schedules, top_scores


def assemble_candidates(picks: List[Schedule]) -> List[MeasureCandidate]:
    """Assemble a list of candidates from a list of schedules

    Parameters
    ----------
    picks: List[tvm.schedule.Schedule]
        The schedules to turn into MeasureCandidates

    Returns
    -------
    measurement_candidates: List[tvm.meta_schedule.MeasureCandidate]
        The list of MeasureCandidates
    """
    return [MeasureCandidate(sch, ArgInfo.from_entry_func(sch.mod, remove_preproc=True)) for sch in picks]


def predict_normalized_scores(schedules: List[Schedule], context: "TuneContext",
                              cost_model: "CostModel") -> List[float]:
    """Predict the normalized score of a list of candidates

    Parameters
    ----------
    schedules: List[tvm.schedule.Schedule]
        The list of schedules to predict scores for
    context: tvm.meta_schedule.TuneContext
        The tune context
    cost_model: tvm.meta_schedule.CostModel
        The cost_model to use for the prediction

    Returns
    -------
    scores: List[float]
        The predicted scores
    """
    assert len(schedules) != 0, "Candidates given for score prediction can not be empty list!"
    scores = cost_model.predict(context, assemble_candidates(schedules))
    return list(scores)


def get_possible_tiling_decisions(tile_product: int, num_tiles: int, max_inntermost_factor=64) -> List[Tuple]:
    """Generates all unique combinations of num_tiles integers whose product equals tile_product

    Parameters
    ----------
    tile_product: int
        The product the tile factors need to multiply to
    num_tiles: int
        The number of tiles (factors)
    max_innermost_factor: int
        The largest innermost tile (factor)

    Returns
    -------
    possible_tiling_decisions: List[Tuple]
        The list of possible tiling decisions
    """
    with Profiler.timeit("BayOptSearch/Tuner/Tune/GetPossibleTilingDecisions"):
        # 1. If the tile_product is equal to one, return a  a list with a single tuple of n 1s
        if tile_product == 1:
            return [(1,) * num_tiles]
        # 2. Catch negative tile products and raise error
        if tile_product < 0:
            raise ValueError(f"The tile product can not be a negative value. Was tile_product = {tile_product}")
        # 3. Catch invalid numer of tiles orr max innerfactor and raise error
        if num_tiles <= 0 or max_inntermost_factor <= 0:
            raise ValueError(f"The number of tiles must be greater than 0. Was num_tiles = {num_tiles},"
                             "max_innermost_factor = {max_inntermost_factor}")

        def factor_combinations(x, start=2, current=[]):
            """Recursively find all factor combinations of x."""
            if x == 1 and len(current) > 0:
                yield current
            else:
                for i in range(start, x + 1):
                    if x % i == 0:
                        yield from factor_combinations(x // i, i, current + [i])

        # 4. Generate all factor combinations that result in tile_product
        all_factors = list(factor_combinations(tile_product))

        # 5. Generate all unique combinations of n elements
        unique_combinations = set()
        for factors in all_factors:
            if len(factors) <= num_tiles:
                padded_factors = factors + [1] * (num_tiles - len(factors))  # Pad with 1s if necessary
                # 6. Generate all permutations of padded_factors to ensure uniqueness
                for perm in set(permutations(padded_factors)):
                    # 7. Ensure the innermost factor is not larger than the max
                    if perm[-1] <= max_inntermost_factor:
                        unique_combinations.add(perm)

        return list(unique_combinations)


def remove_duplicate_schedules(schedules: List[Schedule]) -> List[Schedule]:
    unique_schedules_dict = {str(sch.trace.simplified(remove_postproc=False)): sch for sch in schedules}
    return list(unique_schedules_dict.values())


def get_num_unique_traces(schedules: List[Schedule]):
    """Returns the number unique Traces in a list of Schedules

    Parameters
    ----------
    schedules: List[tvm.schedule.Schedule]
        The list of Schedules

    Returns
    -------
    num: int
        The number of unique Traces in the list of Schedules
    """
    unique_traces = {str(sch.trace.simplified(remove_postproc=False)) for sch in schedules}
    return len(unique_traces)


def create_schedule_from_trace(mod: IRModule, trace: Trace, postprocs: List["Postproc"],
                               rand_state: np.int64, postproc_stats: "PostProcessingStatistic") -> Schedule | None:
    """
    Creates a post processed Schedule from a Trace and IRModule

    Parameters
    ----------
    mod: tvm.ir.IRModule
        The IRModule of the workload
    trace: tvm.schedule.Trace
        The trace to be applied
    postprocs: tvm.meta_schedule.Postproc
        The post-processors
    rand_state: np.int64
        A random state

    Returns
    -------
    sch: tvm.schedule.Schedule | None
        The created schedule or None if post processing fails
    """
    sch = Schedule(mod=mod,
                   seed=rand_state,
                   debug_mask=0,
                   error_render_level="none")
    trace.apply_to_schedule(sch=sch, remove_postproc=True)
    sch.enter_postproc()

    for postproc in postprocs:
        failed = not postproc.apply(sch)
        postproc_stats.enter_result(postproc, failure=failed)
        if failed:
            return None
    return sch


class TuningCandidate:
    """Candidate class for Schedules to be tuned.
    Keeps track of additional information about a Schedule"""
    sch: Schedule = None
    measured: bool = False

    def __init__(self, sch: Schedule, measured: bool) -> None:
        self.sch = sch
        self.measured = measured

    @staticmethod
    def get_schedules(candidates: List["TuningCandidate"]) -> List[Schedule]:
        """Returns the Schedules in the TuningCandidates

        Parameters
        ----------
        candidates: List[TuneCandidate]
            The tuning candidates

        Returns
        -------
        schedules: List[Schedule]
            The list of schedules inside the TuneCandidates
        """
        return [candidate.sch for candidate in candidates]


class TuningReport:
    """Records the tuning progress of a schedule throughout the phases"""
    pre_tuning_score: float = None
    last_tuning_score: float = None
    phase_one_tuning_score: float = None
    phase_two_tuning_score: float = None
    phase_three_tuning_score: float = None

    discarded_tune_schedule: bool = False
    tune_failure: bool = False
    optimizer_failure: bool = False
    num_tuneable_insts: int = None
    num_duplicate_points_skipped: int = 0
    num_points_probed: int = 0

    def __init__(self, measured: bool, is_gpu_target: bool):
        self.measured = measured
        self.is_gpu_target = is_gpu_target

    def create_tuning_result_message(self) -> String:
        """Creates the debug message showing the score throughout the stages"""
        if self.is_gpu_target:
            scores = [self.phase_one_tuning_score]
        else:
            scores = [self.phase_one_tuning_score, self.phase_two_tuning_score, self.phase_three_tuning_score]

        if self.measured:
            message = "(DB) "
        else:
            message = "(RN) "

        if self.pre_tuning_score:
            message += f"{self.pre_tuning_score:.4f} "

        for score in scores:
            if not score:
                message += "==> discarded "
            else:
                message += f"==> {score:.4f} "
        return message

    def analyse_tuning_report(self):
        """Analyses and logs the recorded data"""
        # Due to the use of multiprocessing we need to log results outside of tuner.
        if self.num_tuneable_insts == 0:
            logger(logging.DEBUG, __name__, current_line_number(),
                   "No tuneable decision was found in trace")
        elif self.num_tuneable_insts and self.num_tuneable_insts >= 20:
            logger(logging.WARN, __name__, current_line_number(),
                   "Current workload contains more than 20 tuneable instructions." +
                   "Bayesian Optimization may not be effective.")
        elif self.tune_failure:
            logger(logging.DEBUG, __name__, current_line_number(),
                   "Failed to apply tuning decisions to trace")
        elif self.optimizer_failure:
            logger(logging.ERROR, __name__, current_line_number(),
                   "Optimizer failed to predict next decision")
        else:
            message = self.create_tuning_result_message()
            logger(logging.DEBUG, __name__, current_line_number(), message)


class TuningSummary:
    """Aggregates TuningReports into a summary"""
    improvements: List[float] = []
    best_score: float = 0.0
    num_tune_failures: int = 0
    num_optimizer_failures: int = 0
    num_duplicate_points_skipped: int = 0
    num_points_probed: int = 0
    num_discarded_tune_schedules: int = 0

    def enter_tuning_report(self, tuning_report: TuningReport):
        """Enter a TuningReport into the summary

        Parameters
        ----------
        tuning_report: TuningReport
            The TuningReport to enter
        """
        if tuning_report.tune_failure:
            self.num_tune_failures += 1
        elif tuning_report.optimizer_failure:
            self.num_optimizer_failures += 1
        elif tuning_report.discarded_tune_schedule:
            self.num_discarded_tune_schedules += 1
        elif tuning_report.pre_tuning_score and tuning_report.last_tuning_score:
            self.improvements.append(tuning_report.last_tuning_score - tuning_report.pre_tuning_score)
            if tuning_report.last_tuning_score > self.best_score:
                self.best_score = tuning_report.last_tuning_score

        self.num_duplicate_points_skipped += tuning_report.num_duplicate_points_skipped
        self.num_points_probed += tuning_report.num_points_probed

    def get_avg_improvement(self):
        """Calulates the average improvement over the recorded reports"""
        if len(self.improvements) > 0:
            return sum(self.improvements) / len(self.improvements)
        else:
            return 0

    def log(self):
        """Logs the summary to INFO"""
        logger(logging.INFO, __name__, current_line_number(),
               f"[Tuner] Schedule cost model score improved by an average of: {self.get_avg_improvement():.4f}")
        logger(logging.INFO, __name__, current_line_number(),
               f"[Tuner] Best Score: {self.best_score:.4f}")
        logger(logging.INFO, __name__, current_line_number(),
               f"[Tuner] Number of Tune Failures: {self.num_tune_failures}")
        logger(logging.INFO, __name__, current_line_number(),
               f"[Tuner] Number of Discarded Tuned Schedules: {self.num_discarded_tune_schedules}")
        logger(logging.INFO, __name__, current_line_number(),
               f"[Tuner] Number of Optimizer Failures: {self.num_optimizer_failures}")
        logger(logging.INFO, __name__, current_line_number(),
               f"[Tuner] Number of Duplicate Points Skipped: {self.num_duplicate_points_skipped}")
        logger(logging.INFO, __name__, current_line_number(),
               f"[Tuner] Number of Points Probed: {self.num_points_probed}")


def get_compute_location_insts(sch: Schedule) -> List[Instruction]:
    """Extracts a list of compute location instructions from a Schedule

    Parameters
    ----------
    sch: tvm.schedule.Schedule
        The input schedule

    Returns
    -------
    comp_loc_insts: tvm.tir.Instruction
        The compute location intstructions in the schedule
    """
    compute_location_insts = []

    for inst in sch.trace.insts:
        if inst.kind.name == "SampleComputeLocation":
            compute_location_insts.append(inst)

    return compute_location_insts


class PostProcessingStatistic:
    def __init__(self):
        self.failure_dict: dict = dict()

    def enter_result(self, postproc: "Postproc", failure: bool):
        postproc_name = str(postproc.legacy_repr())
        postproc_name = re.sub(r'\(.*?\)', '', postproc_name)
        self.failure_dict[postproc_name] = self.failure_dict.get(postproc_name, 0) + int(failure)

    def log(self, task_name: str, line_number: int, intro: str):
        """Logs the postprocessing statistics to the task file

        Parameters
        ----------
        task_name: str
            The name of the task
        line_number: int
            The line from which the log was triggered
        intro: str
            Description of what the postprocessing statistic is related to (tuning/sampling)
        """
        # 1. Get the logger
        logger_dict = logging.Logger.manager.loggerDict
        logger_names = list(logger_dict.keys())
        pattern = fr"tvm\.meta_schedule\.logging\.task_\d+_{task_name}$"
        matching_strings = [s for s in logger_names if re.match(pattern, s)]
        assert len(matching_strings) == 1

        # 2. Take the first and only logger, (names are shortened to 100 characters see meta_schedule.logging.py)
        file_logger_obj = get_logger(matching_strings[0][:100])
        file_logger = get_logging_func(file_logger_obj)

        # 3. Create message
        message_lines = [f"{intro}"]
        for index, (postproc_name, num_failures) in enumerate(self.failure_dict.items()):
            line = f"Postproc #{index} {postproc_name}: {num_failures} failure(s)"
            message_lines.append(line)

        # 4. Log Message
        message = "\n".join(message_lines)
        file_logger(logging.INFO, __name__, line_number, message)


def find_file_with_suffix(directory_path, suffix):
    files = os.listdir(directory_path)
    matching_files = [file for file in files if file.endswith(suffix)]
    return matching_files


class BayOptTuner:
    def __init__(self,
                 tune_candidates: List[TuningCandidate],
                 validate_schedules: bool,
                 max_trials: int,
                 optimizer_logging,
                 postprocs,
                 context,
                 cost_model,
                 work_dir,
                 mod,
                 rand_state,
                 only_tune_parallel_extent,
                 is_gpu_target):
        self.tune_candidates: List[TuningCandidate] = tune_candidates
        self.context: TuneContext = context
        self.cost_model: CostModel = cost_model
        self.postprocs: "Postproc" = postprocs
        self.validate_schedules: bool = validate_schedules
        self.max_trials: int = max_trials
        self.optimizer_logging: bool = optimizer_logging
        self.work_dir: str = work_dir
        self.mod: IRModule = mod
        self.rand_state: int = rand_state

        self.log_tuning_traces: bool = False
        self.instruction_decsion_map: dict = dict()
        self.possible_annotate_decisions: dict[str, List[int]] = dict()
        self.path_optimizer_dir: str = self._get_optimizer_dir_path()
        self.optimizer_save_design_space: bool = True
        self.max_optimizer_entries: int = 750
        self.postproc_stats = PostProcessingStatistic()
        self.max_failures: int = 5000
        self.max_sch_failure: int = int(self.max_failures / len(self.tune_candidates))

        self.only_tune_parallel_extent = only_tune_parallel_extent
        self.is_gpu_target = is_gpu_target

        if self.optimizer_logging:
            self._setup_optimizer_dir()

    def tune(self) -> Union[Schedule | TuningReport]:
        tuning_summary = TuningSummary()
        tuned_schedules: List[Schedule] = []
        for candidate in self.tune_candidates:
            # 1. Initialize tuning report
            self.tuning_report: TuningReport = TuningReport(candidate.measured, self.is_gpu_target)
            # 2. Send to tuner
            sch, report = self.tune_single_schedule(candidate.sch, candidate.measured)
            tuned_schedules.append(sch)
            # 3. Analyse report
            report.analyse_tuning_report()
            tuning_summary.enter_tuning_report(report)

        tuning_summary.log()
        self.postproc_stats.log(self.context.task_name, current_line_number(), "Tuning Postproc Summary")
        return tuned_schedules

    def tune_single_schedule(self, untuned_sch: Schedule, measured: bool) -> Union[Schedule | TuningReport]:
        pre_tuning_score = self._predict_normalized_score(untuned_sch)
        self.tuning_report.pre_tuning_score = pre_tuning_score
        self.tuning_report.last_tuning_score = pre_tuning_score

        if self.is_gpu_target:
            with Profiler.timeit("BayOptSearch/Tuner/Tune/BayesianPhase"):
                finished_sch: Schedule = self.bayesian_phase(untuned_sch, measured)
        elif self.only_tune_parallel_extent:
            with Profiler.timeit("BayOptSearch/Tuner/Tune/ParallelPhase"):
                finished_sch: Schedule = self.parallel_phase(untuned_sch)
        else:
            with Profiler.timeit("BayOptSearch/Tuner/Tune/BayesianPhase"):
                sch_phase_one: Schedule = self.bayesian_phase(untuned_sch, measured)
            with Profiler.timeit("BayOptSearch/Tuner/Tune/ParallelPhase"):
                sch_phase_two: Schedule = self.parallel_phase(sch_phase_one)
            with Profiler.timeit("BayOptSearch/Tuner/Tune/ComputeLocationPhase"):
                finished_sch: Schedule = self.compute_location_phase(sch_phase_two)

        return finished_sch, self.tuning_report

    def compute_location_phase(self, sch: Schedule):
        MAX_CANDIDATES = 64

        # 1. Get all compute location insts
        compute_location_insts = get_compute_location_insts(sch)

        # 2. Return if no SampleComputeLocation instructions found
        if len(compute_location_insts) == 0:
            return sch

        # 3. Randomly set an order in which we will work through them
        shuffled_indices = [i for i in range(len(compute_location_insts))]
        random.shuffle(shuffled_indices)

        # 4. Create the first candidates
        first_inst_index = shuffled_indices[0]
        shuffled_indices.pop(0)
        candidates: List[Schedule] = self._get_all_mutations_for_compute_location_insts(sch, first_inst_index)

        # 5. Get all or maximum number of combinations in the shuffled order
        for index in shuffled_indices:
            new_candidates = []
            for candidate in candidates:
                new_candidates.extend(self._get_all_mutations_for_compute_location_insts(candidate, index))
                if (len(candidates) + len(new_candidates)) >= MAX_CANDIDATES:
                    break
            candidates.extend(new_candidates)
            if len(candidates) >= MAX_CANDIDATES:
                break

        # 6. Select the best schedule based on cost model
        if len(candidates) == 0:
            return sch
        else:
            # Get the top schedule and score (returned in lists, be careful)
            top_schs, top_scores = get_top_k_schedules(self.context, self.cost_model, candidates, 0)

            if top_scores[0] <= self.tuning_report.last_tuning_score:
                # If best score worse than before this phase return old schedule
                return sch
            else:
                self.tuning_report.phase_three_tuning_score = top_scores[0]
                self.tuning_report.last_tuning_score = top_scores[0]
                return top_schs[0]

    def _get_all_mutations_for_compute_location_insts(self, sch: Schedule,
                                                      inst_index: int) -> List[Schedule]:
        candidates: List[Schedule] = []
        current_index: int = 0

        for inst in sch.trace.insts:
            if current_index == inst_index and inst.kind.name == "SampleComputeLocation":
                block: BlockRV = inst.inputs[0]
                try:
                    locations: List[IntImm] = collect_compute_location_indices(sch, block)
                    for loc in locations:
                        applied_sch = self._apply_decisions(sch, {inst: loc})
                        if applied_sch is not None:
                            candidates.append(applied_sch)
                        else:
                            self.tuning_report.tune_failure = True
                except Exception:
                    continue
            elif inst.kind.name == "SampleComputeLocation":
                current_index += 1

        return candidates

    def parallel_phase(self, sch: Schedule) -> Schedule:
        possible_decision_dict: Dict[Instruction, List[int]] = dict()

        # Find all parallel annotations and possible decisions
        for inst in sch.trace.insts:
            if (is_annotate_with_parallel(inst)):
                possible_decisions = list(get_possible_parallel_annotate_decisions(
                                                     sch=sch,
                                                     trace=sch.trace,
                                                     rand_state=forkseed(self.rand_state),
                                                     inst=inst,
                                                     max_parallel_extent=16*cpu_count(logical=True)))
                if len(possible_decisions) != 0:
                    possible_decision_dict[inst] = possible_decisions

        # Return if no annotation
        if not bool(possible_decision_dict):
            return sch

        # Prepare all combinations
        schedules: List[Schedule] = []
        for inst, possible_decisions in list(possible_decision_dict.items()):
            for decision in possible_decisions:
                new_sch: Schedule = self._apply_annotation_to_trace(trace=sch.trace,
                                                                    ann_inst=inst,
                                                                    ann_val=decision,
                                                                    mod=self.mod)
                if new_sch is not None:
                    schedules.append(new_sch)

        # Return the best combination
        bests, top_scores = get_top_k_schedules(self.context, self.cost_model, schedules, 1)

        # if top score worse than phase one score, return phase one schedule
        if top_scores[0] <= self.tuning_report.last_tuning_score:
            return sch
        else:
            self.tuning_report.phase_two_tuning_score = top_scores[0]
            self.tuning_report.last_tuning_score = top_scores[0]
            return bests[0]

    def _get_decisions(self, sch: Schedule) -> dict:
        """Retrieves the decision dictionary that identifies the trace and
        can be registered with optimizer

        Parameters
        ----------
        sch: tvm.tir.Schedule
            The schedule of which to generate a decision dict

        Returns
        -------
        decision_dict: dict
            The dictionary containing the decisions or indeces
        """
        input_decisions = dict()

        for inst, decision in sch.trace.decisions.items():
            if inst.kind.name == "SamplePerfectTile":
                # 1. Get the unique tag of the instruction
                inst_dec_tag: str = self._get_parameter_name(inst, decision)
                # 2. Get the key corresponding to all instructions with the same possible decision
                decision_key = self.instruction_decsion_map[inst_dec_tag]
                # 3. Get all possible decisions
                possible_decisions = decision_lookup[decision_key]
                # 4. Get the index of the input decision
                decision_index = possible_decisions.index(tuple(decision))
                # 5. Add index to input decision dictionary
                input_decisions[inst_dec_tag] = float(decision_index)
            elif inst.kind.name == "SampleCategorical":
                # 1. Get the unique tag of the instruction
                inst_dec_tag: str = self._get_parameter_name(inst, decision)
                # 2. The decison is already the required index, so add to dict
                input_decisions[inst_dec_tag] = float(int(decision))

        return input_decisions

    def bayesian_phase(self, untuned_sch: Schedule, measured: bool) -> Schedule:
        pbounds = self._get_parameters(untuned_sch=untuned_sch)

        # Check the number of tuneable instructions
        self.tuning_report.num_tuneable_insts = len(pbounds)
        if self.tuning_report.num_tuneable_insts == 0:
            return untuned_sch

        optimizer = BayesianOptimization(
            f=None,  # We register results with the optimizer ourselves
            pbounds=pbounds,
            verbose=2,
            random_state=forkseed(self.rand_state),
            allow_duplicate_points=False
        )

        discrete_points_registered = dict()
        optimizer = self._configure_optimizer_logging(untuned_sch=untuned_sch, optimizer=optimizer,
                                                      discrete_points_registered=discrete_points_registered)

        kappa = float(os.getenv("TVM_BO_KAPPA", "5"))
        utility = UtilityFunction(kind="ucb", kappa=kappa)

        # Since our input into tuning are schedules with high scores we want to
        # register their decisions with the optimizer, so that it knows about a
        # good result in the beginning.
        input_decisions = self._get_decisions(sch=untuned_sch)
        try:
            optimizer.register(params=input_decisions, target=self.tuning_report.pre_tuning_score)
            discrete_points_registered[str(list(input_decisions.values()))] = None
        except NotUniqueError:
            pass
            # logger(logging.DEBUG, __name__, current_line_number(),
            #        "Duplicate point during optimizer initialization (database schedule)")

        if self.validate_schedules:
            # Validate that recreated trace is identical to input trace
            copy_sch = self._get_schedule_with_predicted_decisons(untuned_sch, input_decisions)
            assert str(untuned_sch.trace) == str(copy_sch.trace)

        max_target: float = 0.0
        max_decisions: dict = None

        current_trial: int = 0
        failure_count: int = 0
        start_time = time.time()
        while (current_trial < self.max_trials and failure_count < self.max_sch_failure
               and time.time() - start_time < 10):
            # Get the a list of decisions for the entered pbounds
            new_decision = False
            next_decisions = None
            iteration = 0
            while not new_decision and iteration < 50:
                iteration += 1
                next_decisions: dict = optimizer.suggest(utility)
                points_to_probe = list(next_decisions.values())
                discrete_points = str([int(x) for x in points_to_probe])
                if discrete_points not in discrete_points_registered:
                    new_decision = True
                    discrete_points_registered[discrete_points] = None
                else:
                    self.tuning_report.num_duplicate_points_skipped += 1

            if next_decisions is None:
                self.tuning_report.optimizer_failure = True
                return untuned_sch

            sch: Schedule = self._get_schedule_with_predicted_decisons(untuned_sch, next_decisions)

            if sch is None:
                failure_count += 1
                continue

            # predict schedule score
            target = self._predict_normalized_score(sch)

            if self.log_tuning_traces:
                logger(logging.INFO, __name__, current_line_number(),
                       f"Target {target} Schedule: \n {sch.trace}\n{sch.mod}")

            # register score with optimizer, to improve next prediction
            try:
                optimizer.register(
                    params=next_decisions,
                    target=target,
                )
            except NotUniqueError as e:
                logger(logging.ERROR, __name__, current_line_number(),
                       f"BO tried to register a duplicate point: {e}")
            # Save best run info
            if target >= max_target or max_decisions is None:
                max_target = target
                max_decisions = next_decisions

            current_trial += 1

        self.tuning_report.num_points_probed = current_trial

        # If the original schedule was never measured (random schedule), and tuning did not improve
        # its score we return the original schedule. However, if we have already measured the schedule
        # (database schedule) then we will measure the worse one instead of measuring the same one twice
        if max_target <= self.tuning_report.pre_tuning_score and not measured:
            self.tuning_report.discarded_tune_schedule = True
            return untuned_sch

        if max_decisions is None:
            self.tuning_report.tune_failure = True
            return untuned_sch

        # Save the tuning score
        self.tuning_report.phase_one_tuning_score = max_target
        self.tuning_report.last_tuning_score = max_target

        # Construct Schedule from best decisions found with BayOpt in this tuning run (not overall)
        tuned_sch: Schedule = self._get_schedule_with_predicted_decisons(untuned_sch, max_decisions)

        if tuned_sch is None:
            self.tuning_report.tune_failure = True
            return untuned_sch

        self._post_tuning_log_copy(tuned_sch=tuned_sch, untuned_sch=untuned_sch)

        if self.log_tuning_traces:
            logger(logging.INFO, __name__, current_line_number(),
                   f"Schedule: \n{tuned_sch.trace}\n{tuned_sch.mod}\n\n\n\n")
        return tuned_sch

    def _get_schedule_with_predicted_decisons(self, untuned_sch: Schedule, next_decisions: dict) -> Schedule | None:
        decisions: Dict[Instruction, DECISION_TYPE] = self._build_decision_dict(untuned_sch, next_decisions)
        tuned_schedule: Schedule | None = self._apply_decisions(untuned_sch, decisions)

        if self.validate_schedules and tuned_schedule is not None:
            self._validate_tuning_decision_application(tuned_schedule, decisions)
        return tuned_schedule

    def _validate_tuning_decision_application(self, sch: Schedule, decisions: Dict[Instruction, DECISION_TYPE]):
        matched_decisions: Dict[Instruction, DECISION_TYPE] = dict()
        for inst, decision in list(decisions.items()):
            matched_inst = self._find_matching_instruction(sch=sch, inst=inst)
            matched_decisions[matched_inst] = decision

        for inst, decision in list(sch.trace.decisions.items()):
            if inst.kind.name == "SamplePerfectTile":
                expected_decision = list(matched_decisions[inst])
                decision = [int(x) for x in decision]

                if expected_decision != decision:
                    logger(logging.ERROR, __name__, current_line_number(),
                           f"Could not find expected decision in trace for {inst} " +
                           f"Expected: {expected_decision} Got: {decision}")

            if inst.kind.name == "SampleCategorical":
                expected_decision = matched_decisions[inst]

                if expected_decision != decision:
                    logger(logging.ERROR, __name__, current_line_number(),
                           f"Could not find expected decision in trace for {inst} " +
                           f"Expected: {expected_decision} Got: {decision}")

    def _apply_decisions(self, untuned_sch: Schedule, decisions: Dict[Instruction, DECISION_TYPE]) -> Schedule | None:
        # Get the schedules trace
        trace: Trace = untuned_sch.trace

        # Apply the decisions to the trace
        for inst, decision in list(decisions.items()):
            trace = trace.with_decision(inst=inst, decision=decision, remove_postproc=True)

        # Create a new schedule from the updated trace and return it
        return create_schedule_from_trace(mod=self.mod, trace=trace, postprocs=self.postprocs,
                                          rand_state=forkseed(self.rand_state),
                                          postproc_stats=self.postproc_stats)

    def _post_tuning_log_copy(self, tuned_sch: Schedule, untuned_sch: Schedule):
        if self.optimizer_logging and not self.optimizer_save_design_space:
            # Save optimizer log with new trace id
            new_trace_id = create_hash(str(tuned_sch.trace))
            file_name: str = f"log_{new_trace_id}.json"
            file_path: str = os.path.join(self.path_optimizer_dir, file_name)

            if not os.path.exists(file_path):
                pre_tuning_trace_id = create_hash(str(untuned_sch.trace))
                pre_tuning_file_name: str = f"log_{pre_tuning_trace_id}.json"
                pre_tuning_file_path: str = os.path.join(self.path_optimizer_dir, pre_tuning_file_name)

                shutil.copy(pre_tuning_file_path, file_path)

    @staticmethod
    def register_points(discrete_points_registered: dict, file_path: str):
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the JSON line
                json_data = json.loads(line)
                decisions = json_data['params']
                points_to_probe = list(decisions.values())
                discrete_points = str([int(x) for x in points_to_probe])
                discrete_points_registered[discrete_points] = None

    def _configure_optimizer_logging(self, untuned_sch: Schedule,
                                     optimizer: BayesianOptimization,
                                     discrete_points_registered: dict) -> BayesianOptimization:
        if not self.optimizer_logging:
            return optimizer
        else:
            if self.optimizer_save_design_space:
                # Each trace is seen without tunable decisions. Will group schedules together which only differ in
                # their tiling and categorical sampling decisions.
                trace_id: str = create_hash(str(
                    BayOptTuner._get_trace_without_tiling_and_categorical_decisions(untuned_sch.trace)))
            else:
                # Each trace (with decisions) is seen as unique
                trace_id: str = create_hash(str(untuned_sch.trace))

            file_path = os.path.join(self.path_optimizer_dir, f"log_{trace_id}_{0}.json")
            newest_file = file_path
            i = 0
            while os.path.exists(newest_file):
                file_path = newest_file
                i += 1
                newest_file = os.path.join(self.path_optimizer_dir, f"log_{trace_id}_{i}.json")

            # if file exists load
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    entries = sum(1 for line in file)

                if self.max_optimizer_entries < entries:
                    file_path = os.path.join(self.path_optimizer_dir, f"log_{trace_id}_{i}.json")
                else:
                    load_logs(optimizer, logs=file_path)
                    BayOptTuner.register_points(discrete_points_registered, file_path)

            # turn logging on again, set reset accordingly
            logger = JSONLogger(path=file_path, reset=False)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            self.optimizer_file_path = file_path
            return optimizer

    @staticmethod
    def _get_trace_without_tiling_and_categorical_decisions(trace: Trace) -> Trace:
        # 1. Get the trace without deadcode and convert to json for easier handling
        trace = trace.simplified(True).as_json()
        # 2. Get the current decisions which are at the end of the json trace
        decisions: List = trace[len(trace) - 1]

        # 3. Now iterate through the instructions and remove the decisions for
        #    tiling and categorical sampling
        index = 0
        for inst in trace[0]:
            if inst[0] == "SamplePerfectTile" or inst[0] == "SampleCategorical":
                decisions.pop(index)
            elif inst[0] == "SampleComputeLocation":
                index += 1

        # 4. Update the decisions in the trace (Note this trace is effectively broken
        #    and should only be used for IDs)
        trace[len(trace) - 1] = decisions
        return trace

    def _setup_optimizer_dir(self):
        if not os.path.exists(self.path_optimizer_dir):
            os.makedirs(self.path_optimizer_dir)

    def _get_optimizer_dir_path(self) -> str:
        work_dir: str = self.work_dir
        return os.path.join(work_dir, "optimizer_logs", self.context.task_name)

    def _find_matching_instruction(self, sch: Schedule, inst: Instruction):
        for new_inst, _ in sch.trace.decisions.items():
            # print(f"{inst.outputs}, {new_inst.outputs}, same {str(new_inst.outputs) == str(inst.outputs)}")
            if str(new_inst.outputs) == str(inst.outputs):
                # print("matched")
                return new_inst

    def _get_parameter_name(self, inst: Instruction, decisions: DECISION_TYPE) -> str:
        name: str = inst.kind.name

        if name == "SamplePerfectTile":
            outputs: str = str(inst.outputs).replace(" ", "")
            n_splits: int = int(inst.attrs[0])
            total_loop_iters: int = int(functools.reduce(operator.mul, decisions))
            return f"{outputs}_{name}_{n_splits}_{total_loop_iters}"
        elif name == "SampleCategorical":
            outputs: str = str(inst.outputs).replace(" ", "")
            return f"{outputs}_{name}"

    def _get_parameters(self, untuned_sch: Schedule):
        pbounds = dict()
        for inst, decisions in untuned_sch.trace.decisions.items():
            # print(inst.outputs, inst, decisions)
            if inst.kind.name == "SamplePerfectTile":
                n_splits: int = int(inst.attrs[0])
                max_innermost_factor: int = int(inst.attrs[1])
                total_loop_iters: int = int(functools.reduce(operator.mul, decisions))

                # Only calculate possible decisions for each pattern once
                decision_key = ("SamplePerfectTile", n_splits, total_loop_iters)
                if decision_key not in decision_lookup:
                    possible_decisions = get_possible_tiling_decisions(total_loop_iters, n_splits,
                                                                       max_innermost_factor)
                    possible_decisions.sort()  # Sort in ascending order, to give the list structure
                    # print(n_splits, total_loop_iters, possible_decisions)
                    decision_lookup[decision_key] = possible_decisions

                inst_dec_tag: str = self._get_parameter_name(inst, decisions)
                pbounds[inst_dec_tag] = (0, len(decision_lookup[decision_key]) - 1)
                self.instruction_decsion_map[inst_dec_tag] = decision_key
            elif inst.kind.name == "SampleCategorical":
                inst_dec_tag: str = self._get_parameter_name(inst, decisions)
                pbounds[inst_dec_tag] = (0, len(inst.attrs[0]) - 1)

        return pbounds

    def _predict_normalized_score(self, sch: Schedule) -> float:
        """Wrapper that allows for score prediction of a single Schedule

        Parameters
        ----------
        sch: tvm.schedule.Schedule
            The schedule to predict a score for

        Returns
        -------
        score: float
            The predicted score
        """
        score = predict_normalized_scores(schedules=[sch],
                                          context=self.context,
                                          cost_model=self.cost_model)
        return score[0]

    def _apply_annotation_to_trace(self, trace: Trace, ann_inst: Instruction,
                                   ann_val: np.int64, mod: IRModule) -> Optional[Schedule]:
        """Allows for the application of an annotation to a trace

        Parameters
        ----------
        trace: tvm.schedule.Trace
            The trace to change the annotation in
        ann_inst: tvm.tir.Instruction
            The annotation Instruction
        ann_val: np.int64
            The value to change the annotation to
        mod: tvm.ir.IRModule
            The IRModule of the workload

        Returns
        -------
        sch: Optional[Schedule]
            Returns schedule with changed annotation if successful
        """
        trace = trace.change_annotation_in_trace(ann_inst, ann_val)

        return create_schedule_from_trace(mod=mod, trace=trace, postprocs=self.postprocs,
                                          rand_state=forkseed(self.rand_state),
                                          postproc_stats=self.postproc_stats)

    def _build_decision_dict(self, untuned_sch: Schedule, next_decisions) -> Dict[Instruction, DECISION_TYPE]:
        result_decisions: Dict[Instruction, DECISION_TYPE] = dict()

        for inst, decisions in untuned_sch.trace.decisions.items():
            if inst.kind.name == "SamplePerfectTile":

                inst_dec_tag: str = self._get_parameter_name(inst, decisions)

                decision_key = self.instruction_decsion_map[inst_dec_tag]
                possible_decisions = decision_lookup[decision_key]

                predicted_index = int(next_decisions[inst_dec_tag])
                result_decisions[inst] = possible_decisions[predicted_index]
            elif inst.kind.name == "SampleCategorical":
                inst_dec_tag: str = self._get_parameter_name(inst, decisions)
                predicted_decision = int(next_decisions[inst_dec_tag])

                tvm_object_decision = make_node("IntImm", dtype=String("int32"), value=predicted_decision, span=None)
                result_decisions[inst] = tvm_object_decision

        return result_decisions


class TuningState:
    """Manages the state and workflow for tuning computational workloads

    This class encapsulates the logic for managing the tuning process of computational workloads, balancing the need
    to explore new schedules with the efficiency of exploiting known high-performing schedules.
    """
    def __init__(self,
                 max_trials: int,
                 num_trials_per_iter: int,
                 design_spaces_schedules: List[Schedule],
                 database: Optional["Database"],
                 cost_model: Optional["CostModel"],
                 rand_state: np.int64,
                 context: "TuneContext",
                 postprocs,
                 population_size,
                 init_min_unmeasured,
                 save_optimizer,
                 max_fail_count,
                 full_first_round_bypass,
                 validate_schedules,
                 is_gpu_target,
                 ):
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_space_schedules = design_spaces_schedules
        self.database: Database = database
        self.cost_model: CostModel = cost_model
        self.rand_state = rand_state
        self.postprocs = postprocs
        self.population_size = population_size
        self.init_min_unmeasured = init_min_unmeasured
        self.save_optimizer = save_optimizer
        self.max_fail_count = max_fail_count
        self.full_first_round_bypass: bool = full_first_round_bypass
        self.validate_schedules: bool = validate_schedules
        self.is_gpu_target: bool = is_gpu_target

        self.context: TuneContext = context
        self.mod: IRModule = context.mod
        self.work_dir: str = self._get_work_dir()

        self.design_spaces: List[Trace] = []
        for space in self.design_space_schedules:
            self.design_spaces.append(space.trace.simplified(True))

        # [st, ed) are the indices of the next batch of candidates.
        self.st: int = 0
        self.ed: int = num_trials_per_iter
        self.bypass_tuning_no_sample_inst: bool = False

        self.workload = database.commit_workload(self.mod)

    def _get_work_dir(self) -> str:
        """Retrieves the working directory path

        Returns
        -------
        path: str
            The working directory path
        """
        path_tuning_record: str = self.database.path_tuning_record
        return os.path.dirname(path_tuning_record)

    def reset(self):
        """Resets the TuningState"""
        self.max_trials = None
        self.num_trials_per_iter = None
        self.design_spaces = None
        self.database = None
        self.cost_model = None

    def _pick_best_from_database(self, num: int) -> List[Schedule]:
        """Retrieves the best schedules for a workload from the database

        Parameters
        ----------
        num: int
            The number of best schedules to return

        Returns
        schedules: List[tvm.schedule.Schedule]
            The list of the best-num of Schedules
        """
        with Profiler.timeit("BayOptSearch/GenerateCandidates/PickBestFromDatabase"):
            # 1. Load top k tuning records for a workload from database
            tuning_records: List[TuningRecord] = self.database.get_top_k(self.workload, num)
            picked_traces: List[Trace] = [record.trace for record in tuning_records]

            # 2. Create Schedules from the traces
            schedules: List[Schedule] = self._process_database_trace(picked_traces)

            logger(logging.INFO, __name__, current_line_number(),
                   f"Picked {len(schedules)} schedules from database")
            return schedules

    def _process_database_trace(self, picked_traces: List[Trace]) -> List[Schedule]:
        """Turns a list of database Traces into Schedules

        Parameters
        ----------
        picked_traces: List[tvm.schedule.Trace]
            The picked Traces

        Returns
        database_schedules: List[tvm.schedule.Schedule]
            The created database schedules
        """
        postproc_stats = PostProcessingStatistic()
        database_schedules: List[Schedule] = []

        for trace in picked_traces:
            sch: Schedule | None = create_schedule_from_trace(mod=self.mod, trace=trace, postprocs=self.postprocs,
                                                              rand_state=forkseed(self.rand_state),
                                                              postproc_stats=postproc_stats)

            if sch is not None:
                database_schedules.append(sch)
            else:
                raise ValueError(f"Could not post-process trace from database:\n{trace}")
        postproc_stats.log(self.context.task_name, current_line_number(), "Database Postproc Summary")
        return database_schedules

    def epsilon_greedy_mix(self, exploit_list: List[Schedule], explore_list: List[Schedule],
                           epsilon: float, num: int, fill_missing: bool) -> List[TuningCandidate]:
        """Mixes exploitation and exploration strategies to select tuning candidates using the epsilon-greedy method.

        Parameters
        -----------
        exploit_list: List[tvm.schedule.Schedule]
            A list of Schedule objects for exploitation (best candidates based on past evaluations).
        explore_list: List[tvm.schedule.Schedule]
            A list of Schedule objects for exploration (potentially good candidates not yet evaluated).
        epsilon: float
            The probability threshold for choosing exploration over exploitation. A higher epsilon values
            increase the likelihood of exploring.
        num: int
            The total number of TuningCandidate objects to return.
        fill_missing: bool
            If True, and if the number of selected candidates is less than 'num', fills the remaining slots
            with candidates from the exploration list.

        Returns
        -------
        mixed_list: List[TuningCandidate]
            A list of TuningCandidate objects selected based on the epsilon-greedy strategy. Each TuningCandidate
            is marked as 'measured' if selected from the exploitation list and 'not measured' if selected from
            the exploration list.

        Background
        ----------
        This method aims to balance the exploration of new schedules with the exploitation of known effective
        schedules. The 'epsilon' parameter controls this balance, with a lower epsilon favoring exploitation and
        a higher epsilon favoring exploration. If 'fill_missing' is True, and the initial mix does not meet the
        'num' criteria, additional exploration candidates are added to ensure a full list of candidates.
        """
        num_explore_schedules = 0
        mixed_list: List[TuningCandidate] = []
        for _ in range(num):
            if random.random() > epsilon:  # Exploitation
                if exploit_list:  # Check if the list is not empty
                    candidate = TuningCandidate(sch=random.choice(exploit_list), measured=True)
                    mixed_list.append(candidate)
            else:  # Exploration
                if explore_list:
                    candidate = TuningCandidate(sch=random.choice(explore_list), measured=False)
                    mixed_list.append(candidate)
                    num_explore_schedules += 1

        # If we don't have measured candidates yet we fill with random
        if fill_missing:
            if len(mixed_list) < num:
                for _ in range(num - len(mixed_list)):
                    candidate = TuningCandidate(sch=random.choice(explore_list), measured=False)
                    mixed_list.append(candidate)
                    num_explore_schedules += 1

            logger(logging.INFO, __name__, current_line_number(),
                   f"Epsilon Greedy mixed {num_explore_schedules} top random schedules into tuning set")
        else:
            logger(logging.INFO, __name__, current_line_number(),
                   f"Epsilon Greedy mixed {len(mixed_list)} random schedules into runner set")
        return mixed_list

    def epsilon_greedy_no_duplicates(self, exploit_list: List[Schedule], explore_list: List[Schedule],
                                     epsilon: float, num: int) -> List[TuningCandidate]:
        """A different approach to mixing the explore and exploit list avoiding duplicates and taking the top
        schedules if the lists are sorted by descending order.

        Parameters
        ----------
        exploit_list: List[tvm.schedule.Schedule]
            The list to pick the exploit Schedules from
        explore_list: List[tvm.schedule.Schedule]
            The list to pick the explore Schedules from
        epsilon: float
            The probability threshold for choosing exploration over exploitation. A higher epsilon values
            increase the likelihood of exploring.
        num: int
            The total number of TuningCandidate objects to return.

        Returns
        -------
        tuning_candidates: List[TuningCandidate]
            The mixed list of TuningCandidates
        """
        if len(exploit_list) == 0:
            num_unmeasured = num
            num_measured = 0
        else:
            num_measured = int(num * (1 - epsilon))
            num_unmeasured = num - num_measured

        selected_measured = [TuningCandidate(sch, True) for sch in exploit_list[:num_measured]]
        selected_unmeasured = [TuningCandidate(sch, False) for sch in explore_list[:num_unmeasured]]

        tune_candidates: List[TuningCandidate] = []
        tune_candidates.extend(selected_measured)
        tune_candidates.extend(selected_unmeasured)

        assert len(tune_candidates) == num

        return tune_candidates

    @staticmethod
    def _has_sample_instruction(traces: List[Trace]) -> bool:
        """
        Returns if a list of traces includes any sample instructions

        Parameters
        ----------
        traces: tvm.schedule.Trace
            The traces to check for sample instructions

        Returns
        -------
        found_sample_inst: bool
            If a sample instruction was found
        """
        # Function could potentially be moved to tvm.schedule.Trace
        sample_instructions = ["SampleComputeLocation", "SampleCategorical", "SamplePerfectTile"]

        for i in range(len(traces)):
            for inst in traces[i].insts:
                if inst.kind.name in sample_instructions:
                    return True
        return False

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """Generates the measurements candidates. This function handles the main control flow

        Returns
        -------
        measurement_candidates: List[tvm.meta_schedule.MeasureCandidate]
            The tuned measurement candidates
        """
        # 1. Check if there are any trials left
        if (self.st >= self.max_trials):
            return None

        # 2. Some workloads have no sample instructions, so we can skip tuning and reduce population
        if not self._has_sample_instruction(traces=self.design_spaces):
            self.num_trials_per_iter = len(self.design_spaces)
            self.bypass_tuning_no_sample_inst = True

        # 3. Check if next batch would go above max trial limit and adjust down
        sample_num = self.num_trials_per_iter
        if (self.ed > self.max_trials):
            sample_num = self.max_trials - self.st
            self.ed = self.max_trials

        assert self.st < self.ed, f"Check failed: {self.st} < {self.ed}"

        # 4. When there are more than 128 entries for a workload in the database
        #    we start mixing the best ones into the tuning set
        num_workload_db_entries = self._get_num_workload_entries()
        measured_schedules: List[Schedule] = []
        if num_workload_db_entries >= 128:
            # Get the top 32 measured schedules in the database
            measured_schedules = self._pick_best_from_database(64)

        # 5. The XGB cost model will give random predictions if the workload does not have
        #    atleast 64 hardware measurement results. Therefore, it can be time efficient to
        #    bypass the tuning stage on the first iter of each workload.
        first_iter_bypass: bool = False
        if (num_workload_db_entries < 64 and self.full_first_round_bypass
                or num_workload_db_entries < self.is_gpu_target):
            first_iter_bypass = True
            logger(logging.INFO, __name__, current_line_number(),
                   "Bypassing BO-Tuner for first 64 Schedules per Workload")

        # 6. Sample a new population of random schedules
        unmeasured_schedules: List[Schedule] = self._sample_initial_population(self.population_size)

        if self.validate_schedules:
            # Gives some insight if the random generation is working as intended
            logger(logging.INFO, __name__, current_line_number(),
                   f"Sampling included {get_num_unique_traces(unmeasured_schedules)} unique schedule(s)")

        # 7. Check if minimum amount of schedules were sampled
        if (len(unmeasured_schedules) < self.init_min_unmeasured):
            raise ValueError("Could not sample a sufficient number of random schedules")

        # 8. Remove duplicates
        unmeasured_schedules = remove_duplicate_schedules(unmeasured_schedules)

        logger(logging.INFO, __name__, current_line_number(),
               f"Prepared a population of {len(measured_schedules) + len(unmeasured_schedules)} " +
               "schedules for selection")

        max_num_tune_schs = len(measured_schedules) + len(unmeasured_schedules)
        sample_num = min(max_num_tune_schs, sample_num)

        # 9. Pick the random and untuned schedules for running (prevent cost model from overfitting)
        random_candidates: List[TuningCandidate] = self.epsilon_greedy_mix(exploit_list=[],
                                                                           explore_list=unmeasured_schedules,
                                                                           epsilon=0.2,
                                                                           num=sample_num,
                                                                           fill_missing=False)

        # 10. Get the best schedules from population
        best_unmeasured_schedules, _ = get_top_k_schedules(self.context, self.cost_model,
                                                           unmeasured_schedules, sample_num)

        # 11. Pick a mix of measured schedules and unmeasured for tuning.
        #     The number of schedules send to the tuner is decided by how many random
        #     schedules were selected for direct measurement.
        # Alternative implementation (contains duplicates)
        # tune_candidates: List[TuningCandidate] = self.epsilon_greedy_mix(exploit_list=measured_schedules,
        #                                                                  explore_list=best_unmeasured_schedules,
        #                                                                  epsilon=0.4,
        #                                                                  num=sample_num - len(random_candidates),
        #                                                                  fill_missing=True)

        tune_candidates: List[TuningCandidate] = self.epsilon_greedy_no_duplicates(
            exploit_list=measured_schedules,
            explore_list=best_unmeasured_schedules,
            epsilon=0.4,
            num=sample_num - len(random_candidates)
        )

        if self.validate_schedules:
            tune_schs = TuningCandidate.get_schedules(tune_candidates)
            # Gives some insight on the number of duplicates entering the tuner
            logger(logging.INFO, __name__, current_line_number(),
                   f"Tuner set includes {get_num_unique_traces(tune_schs)} unique schedule(s)")

        # 12. Sometimes it can make sense to bypass the tuner and prepare the sampled schedules for running immediatley
        #     Possible reasons include: design spaces don't have sample instructions, or first round bypass
        if self.bypass_tuning_no_sample_inst or first_iter_bypass or len(tune_candidates) == 0:
            run_schedules = TuningCandidate.get_schedules(random_candidates) + \
                            TuningCandidate.get_schedules(tune_candidates)
        else:
            # 13. Send the tuning candidates to the tuner
            tuned_schedules: List[Schedule] = self._send_to_bayesian_tuner(tune_candidates)
            run_schedules = TuningCandidate.get_schedules(random_candidates) + tuned_schedules

        assert len(run_schedules) == sample_num
        # 14. Assemble the measurement candidates
        return assemble_candidates(run_schedules)

    def _get_num_workload_entries(self) -> int:
        """Retrieve the number of database entries for a given workload (max = 256)

        Returns
        ----------
        num: int
            The number of entries
        """
        return len(self.database.get_top_k(self.workload, 256))  # ToDo rewrite this properly

    def _send_to_bayesian_tuner(self, tune_candidates: List[TuningCandidate]) -> List[Schedule]:
        """Configures the settings with which the Bayesian Optimizer will tune the schedules, and sends them for tuning.

        Phase Information
        -----------------
        Phase 1: Less than 64 entries in the database for the given workload
            - XGB Cost Model is gives random predicitons
            - Therefore do not save the optimizer
            - Only do one trial (we mostly care about a broader sampling of annotations)

        Phase 2: Less than 256 entries in the database for the given workload
            - XGB Cost Model has become more accurate
            - We can save the optimizer now
            - Set the number of trials to 10 (cost model still not accurate enough to invest more)

        Phase 3: More or equal to 256 entries in the database for the given workload
            - XGB Cost Model is now reliable
            - Increase the number of trials to 20

        Parameters
        ----------
        tune_candidates: List[TuningCandidate]
            The candidate schedules for tuning

        Returns
        -------
        tuned_schedules: List[tvm.schedule.Schedule]
            The list of tuned schedules
        """
        num_workload_db_entries = self._get_num_workload_entries()

        num_trials = 0
        optimizer_logging = False
        only_tune_parallel_extent = False
        if num_workload_db_entries < 64:
            # XGB Cost Model is not yet accurate
            num_trials = 1
            only_tune_parallel_extent = True
        elif num_workload_db_entries < 256:
            num_trials = 10
            optimizer_logging = self.save_optimizer
        else:
            num_trials = 20
            optimizer_logging = self.save_optimizer

        num_sch_to_tuner = len(tune_candidates)
        logger(logging.INFO, __name__, current_line_number(),
               f"Sending {num_sch_to_tuner} schedule(s) to bayesian optimization tuner")

        bo_tuner = BayOptTuner(tune_candidates=tune_candidates,
                               validate_schedules=self.validate_schedules,
                               max_trials=num_trials,
                               optimizer_logging=optimizer_logging,
                               postprocs=self.postprocs,
                               context=self.context,
                               cost_model=self.cost_model,
                               work_dir=self.work_dir,
                               mod=self.mod,
                               rand_state=self.rand_state,
                               only_tune_parallel_extent=only_tune_parallel_extent,
                               is_gpu_target=self.is_gpu_target)
        tuned_schedules = bo_tuner.tune()

        logger(logging.INFO, __name__, current_line_number(), "Bayesian optimization tuner finished")
        return tuned_schedules

    def _sample_initial_population(self, num_schedules: int) -> List[Schedule]:
        """
        Samples an initital population of random schedules, which differ in their decisions

        Parameters
        ----------
        num_schedules: int
            The number of schedules to randomly sample

        Returns
        -------
        schedules: List[tvm.schedule.Schedule]
        """
        with Profiler.timeit("BayOptSearch/GenerateCandidates/SamplePopulation"):
            output_schedules: List[Schedule] = []
            fail_count: int = 0
            postproc_stats = PostProcessingStatistic()
            while (fail_count < self.max_fail_count and
                   len(output_schedules) < self.init_min_unmeasured):

                def create_random_schedule() -> Schedule | None:
                    # 1. Randomly pick a design space
                    design_space_index: int = sample_int(self.rand_state, 0, len(self.design_spaces))
                    # 2. Create a trace with random decisions from design space instructions
                    trace: Trace = Trace(self.design_spaces[design_space_index].insts, {})
                    # 3. Create a schedule from trace
                    sch: Schedule | None = create_schedule_from_trace(mod=self.mod, trace=trace,
                                                                      postprocs=self.postprocs,
                                                                      rand_state=forkseed(self.rand_state),
                                                                      postproc_stats=postproc_stats)
                    return sch

                # 4. Sample random traces
                found_new = False
                for i in range(num_schedules):
                    sch: Schedule | None = create_random_schedule()
                    if (sch is not None):
                        output_schedules.append(sch)
                        found_new = True

                fail_count += int(found_new)
                # 5. Adjust the number of remaining schedules (in case of failures > 0)
                num_schedules -= len(output_schedules)
            postproc_stats.log(self.context.task_name, current_line_number(),
                               "Sample Initial Population Postproc Summary:")
            logger(logging.INFO, __name__, current_line_number(),
                   f"Sampled {len(output_schedules)} new random schedules")
            return output_schedules

    def notify_runner_results(self, measure_candidates: List[MeasureCandidate], results: List[RunnerResult]):
        """This function does not perform any real work currently"""
        self.st += len(results)
        self.ed += len(results)


@derived_object
class BayesianOptimizationSearch(PySearchStrategy):
    context: "TuneContext" = None
    state: TuningState = None

    population_size = 1024  # The number of random schedules sampled
    init_measured_ratio = 0.1
    init_min_unmeasured = 64
    max_fail_count = 10
    save_optimizer: bool = True  # Enables optimizer saving; can be overwritten by optimizer phases
    full_first_round_bypass: bool = False  # Do not tune the first 64 schedules for each workload
    validate_schedules: bool = True  # Use this for debugging; set False for benchmark runs
    is_gpu_target: bool = False

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the search strategy with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initialization.
        """
        self.context: TuneContext = context
        self.postprocs: "Postproc" = context.space_generator.postprocs
        self.rand_state = forkseed(context.rand_state)

        if self.context.target.kind.name == "metal" or self.context.target.kind.name == "cuda":
            self.is_gpu_target = True

    def pre_tuning(
        self,
        max_trials: int,
        num_trials_per_iter: int,
        design_spaces: List[Schedule],
        database: Optional["Database"] = None,
        cost_model: Optional["CostModel"] = None,
    ) -> None:
        """Pre-tuning for the search strategy.

        Parameters
        ----------
        max_trials : int
            The maximum number of trials.
        num_trials_per_iter : int
            The number of trials per iteration.
        design_spaces : List[tvm.tir.Schedule]
            The design spaces used during tuning process.
        database : Optional[Database] = None
            The database used during tuning process.
        cost_model : Optional[CostModel] = None
            The cost model used during tuning process.
        """
        assert design_spaces is not None, "Design spaces should not be None!"
        if self.state is not None:
            raise ValueError("`PreTuning` is already invoked without corresponding `PostTuning`.")

        self.state = TuningState(max_trials=max_trials,
                                 num_trials_per_iter=num_trials_per_iter,
                                 design_spaces_schedules=design_spaces,
                                 database=database,
                                 cost_model=cost_model,
                                 context=self.context,
                                 postprocs=self.postprocs,
                                 rand_state=self.rand_state,
                                 population_size=self.population_size,
                                 init_min_unmeasured=self.init_min_unmeasured,
                                 save_optimizer=self.save_optimizer,
                                 max_fail_count=self.max_fail_count,
                                 full_first_round_bypass=self.full_first_round_bypass,
                                 validate_schedules=self.validate_schedules,
                                 is_gpu_target=self.is_gpu_target)

    def post_tuning(self) -> None:
        """Post-tuning for the search strategy."""
        self.state.reset()

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        """Generate measure candidates from design spaces for measurement.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if finished.
        """
        return self.state.generate_measure_candidates()

    def notify_runner_results(
        self,
        measure_candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> None:
        """Update the search strategy with profiling results.

        Parameters
        ----------
        measure_candidates : List[MeasureCandidate]
            The measure candidates for update.
        results : List[RunnerResult]
            The profiling results from the runner.
        """
        self.state.notify_runner_results(measure_candidates, results)

    def clone(self) -> SearchStrategy:
        """Clone the search strategy.

        Returns
        -------
        strategy : SearchStrategy
            The cloned search strategy.
        """
        return BayesianOptimizationSearch()
