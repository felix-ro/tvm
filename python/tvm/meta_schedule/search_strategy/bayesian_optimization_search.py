from typing import TYPE_CHECKING, List, Optional, Any, Union, Tuple
from tvm.tir.schedule import Schedule, Trace, Instruction, BlockRV
from tvm.tir.analysis import (is_annotate_with_parallel,
                              get_possible_parallel_annotate_decisions,
                              collect_compute_location_indices)
from tvm.ir import IRModule, make_node, structural_hash
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
import re

from bayes_opt import BayesianOptimization, UtilityFunction, SequentialDomainReductionTransformer
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
    return hashlib.md5(input_string.encode()).hexdigest()


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


def get_num_unique_schedules(schedules: List[Schedule]):
    """Returns the number unique programs in a list of Schedules

    Parameters
    ----------
    schedules: List[tvm.schedule.Schedule]
        The list of Schedules

    Returns
    -------
    num: int
        The number of unique Traces in the list of Schedules
    """
    unique_programs = {structural_hash(sch.mod) for sch in schedules}
    return len(unique_programs)


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
        self.measured: bool = measured
        self.is_gpu_target: bool = is_gpu_target

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
    best_score: float = 0.0
    num_tune_failures: int = 0
    num_optimizer_failures: int = 0
    num_duplicate_points_skipped: int = 0
    num_points_probed: int = 0
    num_discarded_tune_schedules: int = 0

    def __init__(self):
        self.improvements: List[float] = []

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
    """Post processors validate if a schedule fulfills all constraints. This class can be used
    to aggregate data on post processing failures."""
    def __init__(self):
        self.failure_dict: dict = dict()

    def enter_result(self, postproc: "Postproc", failure: bool):
        """Enter the postprocessing result into the statistic

        Parameters
        ----------
        postproc: tvm.meta_schedule.Postproc
            The post processor
        failure: bool
            If the post processor failed
        """
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


def find_file_with_suffix(directory_path: str, suffix: str) -> List[str]:
    """Searches for a file in a directory that ends with a given suffix

    Parameters
    ----------
    directory_path: str
        The path of the directory to search in
    suffix: str
        The suffix to search for

    Returns
    -------
    matching_files: List[str]
        A list of file names that match the suffix
    """
    files = os.listdir(directory_path)
    matching_files = [file for file in files if file.endswith(suffix)]
    return matching_files


class BayOptTuner:
    def __init__(self,
                 tune_candidates: List[TuningCandidate],
                 validate_schedules: bool,
                 max_trials: int,
                 optimizer_logging,
                 postprocs: List["Postproc"],
                 context: "TuneContext",
                 cost_model: CostModel,
                 work_dir: str,
                 mod: IRModule,
                 rand_state: int,
                 only_tune_parallel_extent: bool,
                 is_gpu_target: bool,
                 max_optimizer_entries: int,
                 use_sequential_domain_reduction: bool,
                 restricted_memory_logging: bool,
                 acquisition_func_kind: str,
                 kappa: float,
                 xi: float,
                 measured_schedule_hashes: set[int]):
        self.tune_candidates: List[TuningCandidate] = tune_candidates
        self.validate_schedules: bool = validate_schedules
        self.max_trials: int = max_trials
        self.optimizer_logging: bool = optimizer_logging
        self.postprocs: "Postproc" = postprocs
        self.context: TuneContext = context
        self.cost_model: CostModel = cost_model
        self.work_dir: str = work_dir
        self.mod: IRModule = mod
        self.rand_state: int = rand_state
        self.only_tune_parallel_extent = only_tune_parallel_extent
        self.is_gpu_target = is_gpu_target
        self.max_optimizer_entries: int = max_optimizer_entries
        self.use_sequential_domain_reduction: bool = use_sequential_domain_reduction
        self.restricted_memory_logging: bool = restricted_memory_logging
        self.acquisition_func_kind: str = acquisition_func_kind
        self.kappa: float = kappa
        self.xi: float = xi
        self.measured_schedule_hashes = measured_schedule_hashes.copy()

        self.log_tuning_traces: bool = False
        self.instruction_decsion_map: dict = dict()
        self.possible_annotate_decisions: dict[str, List[int]] = dict()
        self.path_optimizer_dir: str = self.get_optimizer_dir_path()
        self.optimizer_save_design_space: bool = True
        self.postproc_stats = PostProcessingStatistic()
        self.max_failures: int = 5000
        self.max_sch_failure: int = int(self.max_failures / len(self.tune_candidates))

        if self.optimizer_logging:
            self.setup_optimizer_dir()

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

        filtered_schedules = []
        for tuned_sch in tuned_schedules:
            if structural_hash(tuned_sch.mod) not in self.measured_schedule_hashes:
                filtered_schedules.append(tuned_sch)

        return filtered_schedules

    def tune_single_schedule(self, untuned_sch: Schedule, measured: bool) -> Union[Schedule | TuningReport]:
        pre_tuning_score = self.predict_normalized_score(untuned_sch)
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
        candidates: List[Schedule] = self.get_all_mutations_for_compute_location_insts(sch, first_inst_index)

        # 5. Get all or maximum number of combinations in the shuffled order
        for index in shuffled_indices:
            new_candidates = []
            for candidate in candidates:
                new_candidates.extend(self.get_all_mutations_for_compute_location_insts(candidate, index))
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

    def get_all_mutations_for_compute_location_insts(self, sch: Schedule,
                                                     inst_index: int) -> List[Schedule]:
        candidates: List[Schedule] = []
        current_index: int = 0

        for inst in sch.trace.insts:
            if current_index == inst_index and inst.kind.name == "SampleComputeLocation":
                block: BlockRV = inst.inputs[0]
                try:
                    locations: List[IntImm] = collect_compute_location_indices(sch, block)
                    for loc in locations:
                        applied_sch = self.apply_decisions(sch, {inst: loc})
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
        possible_decision_dict: dict[Instruction, List[int]] = dict()

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
                new_sch: Schedule = self.apply_annotation_to_trace(trace=sch.trace,
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

    def get_decisions(self, sch: Schedule) -> dict:
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
                inst_dec_tag: str = self.get_parameter_name(inst, decision)
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
                inst_dec_tag: str = self.get_parameter_name(inst, decision)
                # 2. The decison is already the required index, so add to dict
                input_decisions[inst_dec_tag] = float(int(decision))

        return input_decisions

    def get_acquisition_function(self) -> UtilityFunction:
        """Returns the acquisition function

        Returns
        -------
        acq_func: bayes_opt.UtilityFunction
            The configured acquisition function
        """
        if self.acquisition_func_kind == "ucb":
            # Upper Confidence Bound
            acq_func = UtilityFunction(kind="ucb", kappa=self.kappa)
        elif self.acquisition_func_kind == "poi":
            # Probability of Improvement
            acq_func = UtilityFunction(kind="poi", kappa=self.xi)
        elif self.acquisition_func_kind == "ei":
            # Expected Improvement
            acq_func = UtilityFunction(kind="ei", xi=self.xi)
        else:
            raise ValueError(f"Unknown acquisition function of kind: {self.acquisition_func_kind}")
        return acq_func

    def get_next_decision(self, optimizer: BayesianOptimization, acq_func: UtilityFunction,
                          probed_discrete_points: set[str]) -> Optional[dict]:
        new_decision = False
        iteration = 0
        while not new_decision and iteration < 50:
            iteration += 1
            next_decisions: dict = optimizer.suggest(acq_func)
            suggested_decision_values = list(next_decisions.values())
            discrete_decision_points = create_hash(str([int(x) for x in suggested_decision_values]))
            if discrete_decision_points not in probed_discrete_points:
                new_decision = True
                probed_discrete_points.add(discrete_decision_points)
            else:
                self.tuning_report.num_duplicate_points_skipped += 1

        return next_decisions

    def bayesian_phase(self, input_sch: Schedule, measured: bool) -> Schedule:
        """ The Bayesian Optimization phase

        Parameters
        ----------
        input_sch: tvm.schedule.Schedule
            The input Schedule which is used to determine design space
            and optimizer seed
        measured: bool
            If the Schedule is a measured (database) Schedule

        Returns
        -------
        sch: tvm.schedule.Schedule
            The tuned Schedule, the input schedule if unmeasured Schedule
            and no better decisions were recorded or if a failure occurred
        """
        # 1. Extract the parameters from the input schedule
        pbounds = self.get_parameters(untuned_sch=input_sch)

        # 2. Set and check the number of tuneable instructions
        self.tuning_report.num_tuneable_insts = len(pbounds)
        if self.tuning_report.num_tuneable_insts == 0:
            return input_sch

        # 3. Set up sequential domain reduction
        bounds_transformer = None
        if self.use_sequential_domain_reduction:
            bounds_transformer = SequentialDomainReductionTransformer(minimum_window=1)

        # 4. Construct the optimizer
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=forkseed(self.rand_state),
            allow_duplicate_points=False,
            bounds_transformer=bounds_transformer
        )

        # 5. Set up the optimizer logging and read back probed points
        probed_discrete_points: set[str] = set()
        optimizer = self.configure_optimizer_logging(untuned_sch=input_sch, optimizer=optimizer,
                                                     probed_discrete_points=probed_discrete_points)

        # 6. Get the acquisition function (UtilityFunction)
        acq_func: UtilityFunction = self.get_acquisition_function()

        # 7. Since our input into tuning are schedules with high scores we want to register their
        #    decisions with the optimizer, so that it knows about a good point in the beginning.
        input_decisions = self.get_decisions(sch=input_sch)
        try:
            optimizer.register(params=input_decisions, target=self.tuning_report.pre_tuning_score)
            probed_discrete_points.add(create_hash(str(list(input_decisions.values()))))
        except NotUniqueError:
            # When registering a database schedule by hand we may create a duplicate.
            pass

        if self.validate_schedules:
            # Validate that recreated trace is identical to input trace
            copy_sch = self.get_schedule_with_predicted_decisons(input_sch, input_decisions)
            assert structural_hash(input_sch.mod) == structural_hash(copy_sch.mod)

        # ----------------------------------------------------------------------------------------- #
        # 1. Setup is now finished and we can start probing points

        max_score: float = 0.0  # The best score seen during the number of trials
        max_schedule: Optional[Schedule] = None  # The best schedule seen during the number of trials
        current_trial: int = 0  # The current trial (only incremented if schedule passed post processing)
        failure_count: int = 0  # The number of created schedules that failed post processing
        while (current_trial < self.max_trials and failure_count < self.max_sch_failure):
            # 2. Get new decisions to probe
            next_decisions: Optional[dict] = self.get_next_decision(optimizer, acq_func, probed_discrete_points)

            # 3. Check that the optimizer did not fail to suggest a new point
            if next_decisions is None:
                self.tuning_report.optimizer_failure = True
                return input_sch

            # 4. Create schedule based on the predicted decisions
            sch: Optional[Schedule] = self.get_schedule_with_predicted_decisons(input_sch, next_decisions)

            # 5. Check that the schedule passed post processing
            if sch is None:
                failure_count += 1
                continue

            # 6. Get cost model scoring for schedule
            score = self.predict_normalized_score(sch)

            # 7. Register score and decisions with optimizer to improve next suggestion
            try:
                optimizer.register(
                    params=next_decisions,
                    target=score,
                )
            except NotUniqueError as e:
                # 8. We should not get duplicates here
                logger(logging.ERROR, __name__, current_line_number(),
                       f"BO tried to register a duplicate point: {e}")

            # 9. Save the score and schedule if its a new best
            if score >= max_score:
                max_score = score
                max_schedule = sch

            current_trial += 1

        # ----------------------------------------------------------------------------------------- #
        # 1. We have finished probing points let's report what we have done
        self.tuning_report.num_points_probed = current_trial

        # 2. If the original schedule was never measured (random schedule), and tuning did not improve
        #    its score we return the original schedule. However, if we have already measured the schedule
        #    (database schedule) then we will measure the worse one instead of measuring the same one twice
        if max_score <= self.tuning_report.pre_tuning_score and not measured:
            self.tuning_report.discarded_tune_schedule = True
            return input_sch

        # 3. If code validation fails more than self.max_sch_failure we have a tune failure
        if max_schedule is None:
            self.tuning_report.tune_failure = True
            logger(logging.DEBUG, __name__, current_line_number(),
                   f"Experienced {failure_count} failure(s) and did not find a viable schedule")
            return input_sch

        # 4. Save the tuning score
        self.tuning_report.phase_one_tuning_score = max_score
        self.tuning_report.last_tuning_score = max_score
        return max_schedule

    def get_schedule_with_predicted_decisons(self, untuned_sch: Schedule, next_decisions: dict) -> Optional[Schedule]:
        decisions: dict[Instruction, DECISION_TYPE] = self.build_decision_dict(untuned_sch, next_decisions)
        tuned_schedule: Optional[Schedule] = self.apply_decisions(untuned_sch, decisions)

        if self.validate_schedules and tuned_schedule is not None:
            self.validate_tuning_decision_application(tuned_schedule, decisions)
        return tuned_schedule

    def validate_tuning_decision_application(self, sch: Schedule, decisions: dict[Instruction, DECISION_TYPE]):
        matched_decisions: dict[Instruction, DECISION_TYPE] = dict()
        for inst, decision in list(decisions.items()):
            matched_inst = self.find_matching_instruction(sch=sch, inst=inst)
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

    def apply_decisions(self, untuned_sch: Schedule,
                        decisions: dict[Instruction, DECISION_TYPE]) -> Optional[Schedule]:
        # Get the schedules trace
        trace: Trace = untuned_sch.trace

        # Apply the decisions to the trace
        for inst, decision in list(decisions.items()):
            trace = trace.with_decision(inst=inst, decision=decision, remove_postproc=True)

        # Create a new schedule from the updated trace and return it
        return create_schedule_from_trace(mod=self.mod, trace=trace, postprocs=self.postprocs,
                                          rand_state=forkseed(self.rand_state),
                                          postproc_stats=self.postproc_stats)

    def post_tuning_log_copy(self, tuned_sch: Schedule, untuned_sch: Schedule):
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
    def register_discrete_points(discrete_points_registered: set[str], file_path: str):
        """Registers the discrete points probed by reading back the json log

        Parameters
        ----------
        discrete_points_registered: set[str]
            The set containing the probed points as hashes
        file_path: str
            The path to the json log file
        """
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the JSON line
                json_data = json.loads(line)
                decisions = json_data['params']
                points_to_probe = list(decisions.values())
                discrete_points = create_hash(str([int(x) for x in points_to_probe]))
                discrete_points_registered.add(discrete_points)

    def get_most_recent_log_file(self, trace_id: str) -> Union[str, int]:
        """Returns the most recent log file path for a trace_id

        Parameters
        ----------
        trace_id: str
            The trace id for which to find the most recent file

        Returns
        -------
        file_path: str
            The path of the log file
        index: int
            The index number of the file, e.g., '1' for 'log_abcdef_1.json'

        Background
        ----------
        One of the logging methods will start a new file when the logging limit is reached.
        Therefore, it is necessary to find the newest file.
        """
        file_path = os.path.join(self.path_optimizer_dir, f"log_{trace_id}_{0}.json")
        newest_file_path = file_path
        i = 0
        while os.path.exists(newest_file_path):
            file_path = newest_file_path
            i += 1
            newest_file_path = os.path.join(self.path_optimizer_dir, f"log_{trace_id}_{i}.json")
        return file_path, i - 1

    def get_optimizer_trace_id(self, sch: Schedule) -> str:
        """Returns a unique id identifying a unique schedule or group of schedules

        Parameters
        ----------
        sch: tvm.schedule.Schedule
            The schedule for which an ID is needed

        Returns
        -------
        trace_id: str
            The ID of the schedule
        """
        if self.optimizer_save_design_space:
            # Each trace is seen without BO tunable decisions. Will group schedules based on
            # design space, parallel and compute at locations (if they exists (CPU only))
            trace_id: str = create_hash(str(
                BayOptTuner.get_trace_without_tiling_and_categorical_decisions(sch.trace)))
        else:
            # Each trace (with decisions) is seen as unique
            trace_id: str = create_hash(str(sch.trace))

        return trace_id

    @staticmethod
    def remove_first_k_log_entries(file_path: str, k: int):
        """Removes the first k entries in a log file

        Parameters
        ----------
        file_path: str
            The path to the log file
        k: int
            The number of entries to remove
        """
        if os.path.exists(file_path):
            temp_file_path = f"{file_path}.tmp"

            with open(file_path, 'r') as file, open(temp_file_path, 'w') as temp_file:
                for _ in range(k):
                    next(file, None)

                for line in file:
                    temp_file.write(line)

            os.replace(temp_file_path, file_path)

    def configure_optimizer_logging(self, untuned_sch: Schedule,
                                    optimizer: BayesianOptimization,
                                    probed_discrete_points: set[str]) -> BayesianOptimization:
        """Configures the logging behavior of the optimizer

        Parameters
        ----------
        untuned_sch: tvm.schedule.Schedule
            The input schedule of the optimizer
        optimizer: bayes_opt.BayesianOptimization
            The bayesian optimizer to configure
        probed_discrete_points: set[str]
            A dictionary that should contain the discrete points the optimizer has probed

        Returns
        -------
        optimizer: bayes_opt.BayesianOptimization
            The configured optimizer
        """
        # 1. If optimizer logging is turned off do nothing
        if not self.optimizer_logging:
            return optimizer

        # 2. Check if the optimizer groupes schedules
        trace_id: str = self.get_optimizer_trace_id(untuned_sch)

        # 3. Get the corresponding (most recent) log file
        file_path, log_index = self.get_most_recent_log_file(trace_id)

        # 4. If logs exists load them
        if os.path.exists(file_path):
            # 5. Get the number of entries in the file
            with open(file_path, 'r') as file:
                num_entries = sum(1 for line in file)

            # 6. Select the desired logging behavior
            if self.max_optimizer_entries < num_entries and not self.restricted_memory_logging:
                # 7. Start a new log file when limit is reached
                file_path = os.path.join(self.path_optimizer_dir, f"log_{trace_id}_{log_index + 1}.json")
            elif self.max_optimizer_entries < num_entries and self.restricted_memory_logging:
                # 8. Remove the first entries from old log file to make space for new entries
                num_remove = num_entries - self.max_optimizer_entries + self.max_trials
                BayOptTuner.remove_first_k_log_entries(file_path=file_path, k=num_remove)
            else:
                # 9. Continue logging to file
                load_logs(optimizer, logs=file_path)
                BayOptTuner.register_discrete_points(probed_discrete_points, file_path)

        # 10. Finalize settings
        logger = JSONLogger(path=file_path, reset=False)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
        self.optimizer_file_path = file_path
        return optimizer

    @staticmethod
    def get_trace_without_tiling_and_categorical_decisions(trace: Trace) -> Trace:
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

    def setup_optimizer_dir(self):
        if not os.path.exists(self.path_optimizer_dir):
            os.makedirs(self.path_optimizer_dir)

    def get_optimizer_dir_path(self) -> str:
        work_dir: str = self.work_dir
        return os.path.join(work_dir, "optimizer_logs", self.context.task_name)

    def find_matching_instruction(self, sch: Schedule, inst: Instruction):
        for new_inst, _ in sch.trace.decisions.items():
            if str(new_inst.outputs) == str(inst.outputs):
                return new_inst

    def get_parameter_name(self, inst: Instruction, decisions: DECISION_TYPE) -> str:
        name: str = inst.kind.name

        if name == "SamplePerfectTile":
            outputs: str = str(inst.outputs).replace(" ", "")
            n_splits: int = int(inst.attrs[0])
            total_loop_iters: int = int(functools.reduce(operator.mul, decisions))
            return f"{outputs}_{name}_{n_splits}_{total_loop_iters}"
        elif name == "SampleCategorical":
            outputs: str = str(inst.outputs).replace(" ", "")
            return f"{outputs}_{name}"

    def get_parameters(self, untuned_sch: Schedule):
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

                inst_dec_tag: str = self.get_parameter_name(inst, decisions)
                pbounds[inst_dec_tag] = (0, len(decision_lookup[decision_key]) - 1)
                self.instruction_decsion_map[inst_dec_tag] = decision_key
            elif inst.kind.name == "SampleCategorical":
                inst_dec_tag: str = self.get_parameter_name(inst, decisions)
                pbounds[inst_dec_tag] = (0, len(inst.attrs[0]) - 1)

        return pbounds

    def predict_normalized_score(self, sch: Schedule) -> float:
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

    def apply_annotation_to_trace(self, trace: Trace, ann_inst: Instruction,
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

    def build_decision_dict(self, untuned_sch: Schedule, next_decisions) -> dict[Instruction, DECISION_TYPE]:
        result_decisions: dict[Instruction, DECISION_TYPE] = dict()

        for inst, decisions in untuned_sch.trace.decisions.items():
            if inst.kind.name == "SamplePerfectTile":

                inst_dec_tag: str = self.get_parameter_name(inst, decisions)

                decision_key = self.instruction_decsion_map[inst_dec_tag]
                possible_decisions = decision_lookup[decision_key]

                predicted_index = int(next_decisions[inst_dec_tag])
                result_decisions[inst] = possible_decisions[predicted_index]
            elif inst.kind.name == "SampleCategorical":
                inst_dec_tag: str = self.get_parameter_name(inst, decisions)
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
                 search_strategy):
        self.max_trials: int = max_trials
        self.num_trials_per_iter: int = num_trials_per_iter
        self.design_space_schedules: List[Schedule] = design_spaces_schedules
        self.database: Database = database
        self.cost_model: CostModel = cost_model
        self.search_strategy: "PySearchStrategy" = search_strategy

        self.rand_state: int = self.search_strategy.rand_state
        self.postprocs: List["Postproc"] = self.search_strategy.postprocs
        self.context: TuneContext = self.search_strategy.context
        self.mod: IRModule = self.context.mod
        self.work_dir: str = self.get_work_dir()

        self.design_spaces: List[Trace] = []
        for space in self.design_space_schedules:
            self.design_spaces.append(space.trace.simplified(True))

        # [st, ed) are the indices of the next batch of candidates.
        self.st: int = 0
        self.ed: int = num_trials_per_iter
        self.bypass_tuning_no_sample_inst: bool = False

        self.workload = database.commit_workload(self.mod)
        self.measured_schedule_hashes = set()

    def filter_schedules(self, schedules: List[Schedule]) -> List[Schedule]:
        unique_unmeasured_schedules = dict()

        for sch in schedules:
            hashed = structural_hash(sch.mod)
            if hashed not in self.measured_schedule_hashes:
                unique_unmeasured_schedules[hashed] = sch

        return list(unique_unmeasured_schedules.values())

    def get_work_dir(self) -> str:
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

    def register_measured_schedules(self, schedules: List[Schedule]):
        hashed_schedules = [structural_hash(sch.mod) for sch in schedules]
        for hasched_sch in hashed_schedules:
            if hasched_sch not in self.measured_schedule_hashes:
                self.measured_schedule_hashes.add(hasched_sch)

    def get_schedules_from_database(self) -> List[Schedule]:
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
            # 1. Load all tuning records for a workload from database
            tuning_records: List[TuningRecord] = self.database.get_top_k(self.workload, len(self.database))
            picked_traces: List[Trace] = [record.trace for record in tuning_records]

            # 2. Create Schedules from the traces
            schedules: List[Schedule] = self.process_database_trace(picked_traces)

            # 3. Register measured schedules to avoid measuring them again
            self.register_measured_schedules(schedules)

            logger(logging.INFO, __name__, current_line_number(),
                   f"Picked {get_num_unique_schedules(schedules)} schedules from database")
            return schedules

    def process_database_trace(self, picked_traces: List[Trace]) -> List[Schedule]:
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
            if random.random() > epsilon:
                # Pick exploitation schedule
                if len(exploit_list) > 0:
                    index = sample_int(self.rand_state, 0, len(exploit_list))
                    candidate = TuningCandidate(sch=exploit_list[index], measured=True)
                    exploit_list.pop(index)
                    mixed_list.append(candidate)
            else:
                # Pick exploration schedule
                if len(explore_list) > 0:
                    index = sample_int(self.rand_state, 0, len(explore_list))
                    candidate = TuningCandidate(sch=explore_list[index], measured=False)
                    explore_list.pop(index)
                    mixed_list.append(candidate)
                    num_explore_schedules += 1

        # If we don't have measured candidates yet we fill with random
        if fill_missing:
            if len(mixed_list) < num:
                for _ in range(num - len(mixed_list)):
                    index = sample_int(self.rand_state, 0, len(explore_list))
                    candidate = TuningCandidate(sch=explore_list[index], measured=False)
                    explore_list.pop(index)
                    mixed_list.append(candidate)
                    num_explore_schedules += 1

            logger(logging.INFO, __name__, current_line_number(),
                   f"Epsilon Greedy mixed {num_explore_schedules} top random schedules into tuning set")
        else:
            logger(logging.INFO, __name__, current_line_number(),
                   f"Epsilon Greedy mixed {len(mixed_list)} random schedules into runner set")
        return mixed_list

    def epsilon_greedy_only_top(self, exploit_list: List[Schedule], explore_list: List[Schedule],
                                epsilon: float, num: int) -> List[TuningCandidate]:
        """A different approach to mixing the explore and exploit list taking the top
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

        # assert len(tune_candidates) == num

        return tune_candidates

    @staticmethod
    def has_sample_instruction(traces: List[Trace]) -> bool:
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
        if not self.has_sample_instruction(traces=self.design_spaces):
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
        measured_schedules: List[Schedule] = self.get_schedules_from_database()
        num_workload_db_entries = len(measured_schedules)

        # 5. The XGB cost model will give random predictions if the workload does not have
        #    atleast 64 hardware measurement results. Therefore, it can be time efficient to
        #    bypass the tuning stage on the first iter of each workload.
        first_iter_bypass: bool = False
        if (num_workload_db_entries < 64 and self.search_strategy.full_first_round_bypass
                or num_workload_db_entries < 64 and self.search_strategy.is_gpu_target):
            first_iter_bypass = True
            logger(logging.INFO, __name__, current_line_number(),
                   "Bypassing BO-Tuner for first 64 Schedules per Workload")

        # 6. Sample a new population of random schedules
        unmeasured_schedules: List[Schedule] = self.sample_initial_population(self.search_strategy.population_size)

        if self.search_strategy.validate_schedules:
            # Gives some insight if the random generation is working as intended
            logger(logging.INFO, __name__, current_line_number(),
                   f"Sampling included {get_num_unique_schedules(unmeasured_schedules)} unique schedule(s)")

        # 7. Check if minimum amount of schedules were sampled
        if (len(unmeasured_schedules) < self.search_strategy.init_min_unmeasured):
            raise ValueError("Could not sample a sufficient number of random schedules")

        # 8. Remove duplicates
        unmeasured_schedules = self.filter_schedules(unmeasured_schedules)

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

        # Register the random candidates already
        self.register_measured_schedules(TuningCandidate.get_schedules(random_candidates))

        # 10. Get the best schedules from population
        best_unmeasured_schedules = []
        if (len(unmeasured_schedules) > 0):
            best_unmeasured_schedules, _ = get_top_k_schedules(self.context, self.cost_model,
                                                               unmeasured_schedules, sample_num)

        tune_candidates: List[TuningCandidate] = self.epsilon_greedy_only_top(
            exploit_list=measured_schedules,
            explore_list=best_unmeasured_schedules,
            epsilon=0.4,
            num=sample_num - len(random_candidates)
        )

        if self.search_strategy.validate_schedules:
            tune_schs = TuningCandidate.get_schedules(tune_candidates)
            # Gives some insight on the number of duplicates entering the tuner
            logger(logging.INFO, __name__, current_line_number(),
                   f"Tuner set includes {get_num_unique_schedules(tune_schs)} unique schedule(s)")

        # 12. Sometimes it can make sense to bypass the tuner and prepare the sampled schedules for running immediatley
        #     Possible reasons include: design spaces don't have sample instructions, or first round bypass
        if self.bypass_tuning_no_sample_inst or first_iter_bypass or len(tune_candidates) == 0:
            run_schedules = TuningCandidate.get_schedules(random_candidates) + \
                            TuningCandidate.get_schedules(tune_candidates)
        else:
            # 13. Send the tuning candidates to the tuner
            tuned_schedules: List[Schedule] = self.send_to_bayesian_tuner(tune_candidates)
            run_schedules = TuningCandidate.get_schedules(random_candidates) + tuned_schedules

        if self.search_strategy.validate_schedules:
            logger(logging.INFO, __name__, current_line_number(),
                   f"Runner set includes {get_num_unique_schedules(run_schedules)} unique schedule(s)")
        # assert len(run_schedules) == sample_num
        # 14. Assemble the measurement candidates
        return assemble_candidates(run_schedules)

    def get_num_workload_entries(self) -> int:
        """Retrieve the number of database entries for a given workload (max = 256)

        Returns
        ----------
        num: int
            The number of entries
        """
        return len(self.database.get_top_k(self.workload, 256))  # ToDo rewrite this properly

    def send_to_bayesian_tuner(self, tune_candidates: List[TuningCandidate]) -> List[Schedule]:
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
        num_workload_db_entries = self.get_num_workload_entries()

        num_trials = 0
        optimizer_logging = False
        only_tune_parallel_extent = False
        if num_workload_db_entries < 64:
            # XGB Cost Model is not yet accurate
            num_trials = 1
            only_tune_parallel_extent = True
        elif num_workload_db_entries < 256:
            num_trials = 10
            optimizer_logging = self.search_strategy.save_optimizer
        else:
            num_trials = 20
            optimizer_logging = self.search_strategy.save_optimizer

        num_sch_to_tuner = len(tune_candidates)
        logger(logging.INFO, __name__, current_line_number(),
               f"Sending {num_sch_to_tuner} schedule(s) to bayesian optimization tuner")

        bo_tuner = BayOptTuner(tune_candidates=tune_candidates,
                               validate_schedules=self.search_strategy.validate_schedules,
                               max_trials=num_trials,
                               optimizer_logging=optimizer_logging,
                               postprocs=self.postprocs,
                               context=self.context,
                               cost_model=self.cost_model,
                               work_dir=self.work_dir,
                               mod=self.mod,
                               rand_state=self.rand_state,
                               only_tune_parallel_extent=only_tune_parallel_extent,
                               is_gpu_target=self.search_strategy.is_gpu_target,
                               max_optimizer_entries=self.search_strategy.max_optimizer_entries,
                               use_sequential_domain_reduction=self.search_strategy.use_sequential_domain_reduction,
                               restricted_memory_logging=self.search_strategy.restricted_memory_logging,
                               acquisition_func_kind=self.search_strategy.acquisition_func_kind,
                               kappa=self.search_strategy.kappa,
                               xi=self.search_strategy.xi,
                               measured_schedule_hashes=self.measured_schedule_hashes)
        tuned_schedules = bo_tuner.tune()

        logger(logging.INFO, __name__, current_line_number(), "Bayesian optimization tuner finished")
        return tuned_schedules

    def sample_initial_population(self, num_schedules: int) -> List[Schedule]:
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
            while (fail_count < self.search_strategy.max_fail_count and
                   len(output_schedules) < self.search_strategy.init_min_unmeasured):

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

    def __init__(self):
        self.max_optimizer_entries: int = int(os.getenv("TVM_BO_MAX_OPTIMIZER_ENTRIES", "500"))
        self.use_sequential_domain_reduction: bool = (
            os.getenv("TVM_BO_USE_SEQUENTIAL_DOMAIN_REDUCTION", "False") == "True"
        )
        self.restricted_memory_logging: bool = os.getenv("TVM_BO_RESTRICTED_MEMORY_LOGGING", "False") == "True"
        self.acquisition_func_kind: str = os.getenv("TVM_BO_ACQUISITION_FUNCTION", "ucb")
        self.kappa: float = float(os.getenv("TVM_BO_KAPPA", "5"))
        self.xi: float = float(os.getenv("TVM_BO_XI", "0.1"))

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
                                 search_strategy=self)

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
