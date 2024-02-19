from typing import TYPE_CHECKING, List, Optional, Any, Dict
from tvm.tir.schedule import Schedule, Trace, Instruction
from tvm.ir import IRModule, make_node
from tvm.runtime import String

from .search_strategy import PySearchStrategy, MeasureCandidate, SearchStrategy
from ..utils import derived_object
from ..arg_info import ArgInfo
from ..runner import RunnerResult
from ..logging import get_logger, get_logging_func
from ..profiler import Profiler

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database, TuningRecord
    from ..tune_context import TuneContext

    ATTR_TYPE = Any

import numpy as np
import copy
import random
import functools
import operator
import logging
import inspect
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import shutil


from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


DECISION_TYPE = Any


decision_lookup = dict()


def create_hash(input_string: str) -> str:
    input_bytes = input_string.encode('utf-8')
    hash_obj = hashlib.sha256(input_bytes)
    hash_hex: str = hash_obj.hexdigest()

    return hash_hex


def current_line_number():
    return inspect.currentframe().f_back.f_lineno


def forkseed(rand_state):
    rand_state = int(rand_state+random.random()*1999999973)
    new_rand_state = (rand_state * 32767) % 1999999973
    return new_rand_state


# min_inclusive & max_exclusive: [min, max)
def sample_int(rand_state: np.int64, min_inclusive: int, max_exclusive: int):
    assert min_inclusive < max_exclusive, "ValueError: max_exclusive must be greater than min_inclusive."

    if ((min_inclusive + 1) == max_exclusive):
        return min_inclusive
    rand_ = forkseed(rand_state)
    # call np.random to generate [min, max-1]
    np.random.seed(rand_)
    dist = random.randint(min_inclusive, max_exclusive-1)
    return dist


def assemble_candidates(picks: List[Schedule]) -> List[MeasureCandidate]:
    """Assemble a list of candidates from a list of schedules."""
    measure_inputs = [None for _ in range(len(picks))]
    for i, sch in enumerate(picks):
        measure_inputs[i] = MeasureCandidate(sch, ArgInfo.from_entry_func(sch.mod, remove_preproc=True))
    return measure_inputs


def predict_normalized_scores(candidates, context, cost_model):
    """Predict the normalized score of a list of candidates."""
    assert len(candidates) != 0, "Candidates given for score prediction can not be empty list!"
    scores = cost_model.predict(context, assemble_candidates(candidates))

    # print(f"TLP predict = {scores}")
    scores = np.clip(scores, 0.0, np.inf)
    # print(f"TLP predict normal score = {scores}")
    return scores


def get_possible_tiling_decisions(tile_product, num_tiles):
    """Generates all unique combinations of num_tiles integers whose product equals tile_product"""
    with Profiler.timeit("BayOptSearch/Tuner/Tune/GetPossibleTilingDecisions"):
        # Special case handling for x=1
        if tile_product == 1:
            return [(1,) * num_tiles]  # Return a list with a single tuple of n 1s

        # Base case handling
        if tile_product < 0:
            return "No solution for negative X with only positive integers."
        if num_tiles <= 0:
            return "Invalid n. n must be greater than 0."

        def factor_combinations(x, start=2, current=[]):
            """Recursively find all factor combinations of x."""
            if x == 1 and len(current) > 0:
                yield current
            else:
                for i in range(start, x + 1):
                    if x % i == 0:
                        yield from factor_combinations(x // i, i, current + [i])

        # Generate all factor combinations
        all_factors = list(factor_combinations(tile_product))

        # Generate all unique combinations of n elements
        unique_combinations = set()
        for factors in all_factors:
            if len(factors) <= num_tiles:
                padded_factors = factors + [1] * (num_tiles - len(factors))  # Pad with 1s if necessary
                # Generate all permutations of padded_factors to ensure uniqueness
                for perm in set(permutations(padded_factors)):
                    unique_combinations.add(perm)

        return list(unique_combinations)


def process_database_trace(trace_id, per_thread_data, picked_traces, pp: "ThreadedTraceApply", results, num_threads):
    thread_id = trace_id % num_threads
    data = per_thread_data[thread_id]
    rand_state = data.rand_state
    mod = data.mod

    trace = picked_traces[trace_id]
    result = results[trace_id]
    assert result is None, f"result {trace_id} should be None"

    sch: Schedule = pp.apply(mod=mod, trace=trace, rand_state=rand_state)

    if sch is not None:
        results[trace_id] = sch
    else:
        raise ValueError(f"Could not post-process trace from database:\n{trace}")


class ThreadedTraceApply:
    class Item:
        postproc = None
        fail_counter = 0

        def __init__(self, postproc):
            self.postproc = postproc

    def __init__(self, postprocs) -> None:
        self.n_ = len(postprocs)
        self.items_ = [self.Item(postprocs[i]) for i in range(self.n_)]

    def apply(self, mod: IRModule, trace: Trace, rand_state: np.int64) -> (Schedule | None):
        sch = Schedule(mod=mod,
                       seed=forkseed(rand_state),
                       debug_mask=0,
                       error_render_level="none",)
        trace.apply_to_schedule(sch=sch, remove_postproc=True)
        sch.enter_postproc()

        for i in range(self.n_):
            item = self.items_[i]
            if not item.postproc.apply(sch):
                item.fail_counter += 1
                return None
        return sch


class PerThreadData:
    mod: IRModule = None
    rand_state: np.int64 = np.int64(-1)

    def __init__(self) -> None:
        self.mod = None
        self.rand_state = np.int64(-1)


class BayOptTuner:
    def __init__(self,
                 sch: Schedule,
                 state: "State",
                 save_optimizer: bool,
                 validate_schedules: bool):
        self.sch: Schedule = sch
        self.postprocs = state.search_strategy.postprocs
        self.state: State = state
        self.save_optimizer: bool = save_optimizer
        self.validate_schedules: bool = validate_schedules

        self.context: TuneContext = state.context
        self.cost_model: CostModel = state.cost_model
        self.max_trials: int = 10
        self.instruction_decsion_map = dict()
        self.path_optimizer_dir: str = self._get_optimizer_dir_path()

    def tune(self) -> Schedule:
        pre_tuning_score = self._predict_normalized_score(self.sch)
        pbounds = self._get_parameters()

        # Check if there are any tunable instructions
        if len(pbounds) == 0:
            self.state.logger(logging.DEBUG, __name__, current_line_number(),
                              "No tuneable decision was found in trace")
            return self.sch

        optimizer = BayesianOptimization(
            f=None,  # We register results with the optimizer ourselves
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )

        optimizer = self._configure_optimizer_logging(optimizer)
        utility = UtilityFunction(kind="ucb", kappa=5, xi=0.0)

        current_trial: int = 0
        while (current_trial < self.max_trials):
            # Get the a list of decisions for the entered pbounds
            next_decisions: dict = optimizer.suggest(utility)
            sch: Schedule = self._get_schedule_with_predicted_decisons(next_decisions)

            if sch is None:
                self.state.logger(logging.DEBUG, __name__, current_line_number(),
                                  "Failed to apply tuning decisions to trace")
                continue

            # predict schedule score
            target = self._predict_normalized_score(sch)

            # register score with optimizer, to improve next prediction
            optimizer.register(
                params=next_decisions,
                target=target,
            )
            current_trial += 1

        # Get the best decisions construct schedule again
        post_tuning_score = optimizer.max['target']
        self.state.logger(logging.DEBUG, __name__, current_line_number(),
                          f"Pre tuning score: {pre_tuning_score:.4f} ==> Post tuning score: {post_tuning_score:.4f}")

        # If worse than untuned, return original schedule
        if (post_tuning_score < pre_tuning_score):
            return self.sch

        # Construct Schedule from best decisions found with BayOpt
        best_decisions = optimizer.max["params"]
        tuned_sch: Schedule = self._get_schedule_with_predicted_decisons(best_decisions)

        self.state.logger(logging.INFO, __name__, current_line_number(), f"\n{tuned_sch.trace}")

        if tuned_sch is None:
            self.state.logger(logging.DEBUG, __name__, current_line_number(),
                              "Failed to apply tuning decisions to trace")
            return self.sch

        self._post_tuning_log_copy(tuned_sch)
        return tuned_sch

    def _get_schedule_with_predicted_decisons(self, next_decisions: dict) -> Schedule:
        decisions: Dict[Instruction, DECISION_TYPE] = self._build_decision_dict(next_decisions)
        tuned_schedule: Schedule = self._apply_decisions(decisions)

        if self.validate_schedules:
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
                    self.state.logger(logging.ERROR, __name__, current_line_number(),
                                      f"Could not find expected decision in trace. \
                                        Expected: {expected_decision} Got: {decision}")

    def _apply_decisions(self, decisions: Dict[Instruction, DECISION_TYPE]) -> Schedule:
        data: PerThreadData = self.state.per_thread_data_[0]
        sch: Schedule = self.sch
        mod: IRModule = data.mod

        # Apply the decisions to the trace
        for inst, decision in list(decisions.items()):
            # Retracing the graph changes instruction handles
            # We need to find instruction in trace after each iteration
            matched_inst = self._find_matching_instruction(sch=sch, inst=inst)

            sch: Schedule = self._apply_decision_to_trace(mod=mod,
                                                          trace=sch.trace,
                                                          inst=matched_inst,
                                                          decision=decision)
        return sch

    def _post_tuning_log_copy(self, sch: Schedule):
        if self.save_optimizer:
            # Save optimizer log with new trace id
            new_trace_id = create_hash(str(sch.trace))
            file_name: str = f"log_{new_trace_id}.json"
            file_path: str = os.path.join(self.path_optimizer_dir, file_name)

            if not os.path.exists(file_path):
                pre_tuning_trace_id = create_hash(str(self.sch.trace))
                pre_tuning_file_name: str = f"log_{pre_tuning_trace_id}.json"
                pre_tuning_file_path: str = os.path.join(self.path_optimizer_dir, pre_tuning_file_name)

                shutil.copy(pre_tuning_file_path, file_path)

    def _configure_optimizer_logging(self, optimizer: BayesianOptimization) -> BayesianOptimization:
        if not self.save_optimizer:
            return optimizer
        else:
            self._setup_optimizer_dir()

            # give each trace a unique name
            trace_id: str = create_hash(str(self.sch.trace))
            file_name: str = f"log_{trace_id}.json"
            file_path: str = os.path.join(self.path_optimizer_dir, file_name)

            # if file exists load
            if os.path.exists(file_path):
                load_logs(optimizer, logs=file_path)

            # turn logging on again, set reset accordingly
            logger = JSONLogger(path=file_path, reset=False)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            return optimizer

    def _setup_optimizer_dir(self) -> None:
        directory = os.path.dirname(self.path_optimizer_dir)

        if directory and not os.path.exists(self.path_optimizer_dir):
            os.mkdir(self.path_optimizer_dir)

    def _get_optimizer_dir_path(self) -> str:
        work_dir: str = self.state.work_dir
        return os.path.join(work_dir, "optimizer_logs")

    @staticmethod
    def _find_matching_instruction(sch: Schedule, inst: Instruction):
        for new_inst, _ in sch.trace.decisions.items():
            # print(f"{inst.outputs}, {new_inst.outputs}, same {str(new_inst.outputs) == str(inst.outputs)}")
            if str(new_inst.outputs) == str(inst.outputs):
                # print("matched")
                return new_inst

    @staticmethod
    def _get_parameter_name(inst: Instruction, decisions: DECISION_TYPE) -> str:
        name: str = inst.kind.name

        if name == "SamplePerfectTile":
            outputs: str = str(inst.outputs).replace(" ", "")
            n_splits: int = int(inst.attrs[0])
            total_loop_iters: int = int(functools.reduce(operator.mul, decisions))

            return f"{outputs}_{name}_{n_splits}_{total_loop_iters}"
        elif name == "SampleCategorical":
            outputs: str = str(inst.outputs).replace(" ", "")
            return f"{outputs}_{name}"

    def _get_parameters(self):
        pbounds = dict()
        # self.sch.trace.show()
        for inst, decisions in self.sch.trace.decisions.items():
            # print(inst.outputs, inst, decisions)
            if inst.kind.name == "SamplePerfectTile":
                n_splits: int = int(inst.attrs[0])
                total_loop_iters: int = int(functools.reduce(operator.mul, decisions))

                # Only calculate possible decisions for each pattern once
                decision_key = ("SamplePerfectTile", n_splits, total_loop_iters)
                if decision_key not in decision_lookup:
                    possible_decisions = get_possible_tiling_decisions(total_loop_iters, n_splits)
                    possible_decisions.sort()  # Sort in ascending order, to give the list structure
                    # print(n_splits, total_loop_iters, possible_decisions)
                    decision_lookup[decision_key] = possible_decisions

                inst_dec_tag: str = self._get_parameter_name(inst, decisions)
                pbounds[inst_dec_tag] = (0, len(decision_lookup[decision_key]) - 1)
                self.instruction_decsion_map[inst_dec_tag] = decision_key
            elif inst.kind.name == "SampleCategorical":
                inst_dec_tag: str = self._get_parameter_name(inst, decisions)
                print(len(inst.attrs[0]) - 1)
                print(decisions, type(decisions))
                pbounds[inst_dec_tag] = (0, len(inst.attrs[0]) - 1)

        return pbounds

    def _predict_normalized_score(self, sch: Schedule) -> float:
        score = predict_normalized_scores(candidates=[sch],
                                          context=self.context,
                                          cost_model=self.cost_model)
        return score[0]

    def _apply_decision_to_trace(self, mod: IRModule, trace: Trace,
                                 inst: Instruction, decision: DECISION_TYPE) -> Schedule | None:

        if inst.kind.name == "SampleCategorical":
            print(inst)
            print(decision)

        trace = trace.with_decision(inst=inst, decision=decision, remove_postproc=True)

        if inst.kind.name == "SampleCategorical":
            print("success")

        pp = ThreadedTraceApply(postprocs=self.postprocs)
        return pp.apply(mod=mod, trace=trace, rand_state=1)

    def _build_decision_dict(self, next_decisions) -> Dict[Instruction, DECISION_TYPE]:
        result_decisions: Dict[Instruction, DECISION_TYPE] = dict()

        for inst, decisions in self.sch.trace.decisions.items():
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


class State:
    def __init__(self,
                 max_trials: int,
                 num_trials_per_iter: int,
                 design_spaces_schedules: List[Schedule],
                 database: Optional["Database"],
                 cost_model: Optional["CostModel"],
                 search_strategy: "BayesianOptimizationSearch",
                 context: "TuneContext"
                 ):
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_space_schedules = design_spaces_schedules
        self.database: Database = database
        self.cost_model: CostModel = cost_model
        self.search_strategy: BayesianOptimizationSearch = search_strategy
        self.context: TuneContext = context
        self.mod = context.mod
        self.logger = get_logging_func(get_logger(__name__))
        self.work_dir: str = self._get_work_dir()

        self.design_spaces = []
        for space in self.design_space_schedules:
            self.design_spaces.append(space.trace.simplified(True))

        # [st, ed) are the indices of the next batch of candidates.
        self.st: int = 0
        self.ed: int = num_trials_per_iter
        self.max_fail_count: int = 300

        self.per_thread_data_ = [PerThreadData() for i in range(self.context.num_threads)]
        for i in range(self.context.num_threads):
            self.per_thread_data_[i].mod = copy.deepcopy(self.mod)
            self.per_thread_data_[i].rand_state = forkseed(self.search_strategy.rand_state)

        self.workload = database.commit_workload(self.mod)

    def _get_work_dir(self) -> str:
        path_tuning_record: str = self.database.path_tuning_record
        return os.path.dirname(path_tuning_record)

    def reset(self):
        self.max_trials = None
        self.num_trials_per_iter = None
        self.design_spaces = None
        self.database = None
        self.cost_model = None

    def _pick_best_from_database(self, num: int) -> List[Schedule]:
        with Profiler.timeit("BayOptSearch/GenerateCandidates/PickBestFromDatabase"):
            picked_traces: List[Trace] = []
            # Load top k tuning records for a workload from database
            tuning_records: List[TuningRecord] = self.database.get_top_k(self.workload, num)

            for record in tuning_records:
                picked_traces.append(record.trace)

            # Get the actual number of picked traces, (there may have not been enough in the database)
            actual_num_picked = len(picked_traces)
            pp: ThreadedTraceApply = ThreadedTraceApply(self.search_strategy.postprocs)

            results: List[Schedule] = [None] * actual_num_picked
            for i in range(actual_num_picked):
                process_database_trace(i, self.per_thread_data_, picked_traces, pp, results, self.context.num_threads)
            self.logger(logging.INFO, __name__, current_line_number(),
                        f"Picked {len(results)} schedules from database")
            return results

    def epsilon_greedy_mix(self, exploit_list, explore_list, epsilon, num, for_tuning: bool):
        num_explore_schedules = 0
        mixed_list = []
        for _ in range(num):
            if random.random() > epsilon:  # Exploitation
                if exploit_list:  # Check if the list is not empty
                    mixed_list.append(random.choice(exploit_list))
            else:  # Exploration
                if explore_list:
                    mixed_list.append(random.choice(explore_list))
                    num_explore_schedules += 1

        # If we don't have measured candidates yet we fill with random
        if for_tuning:
            if len(mixed_list) < num:
                for _ in range(num - len(mixed_list)):
                    mixed_list.append(random.choice(explore_list))
                    num_explore_schedules += 1

            self.logger(logging.INFO, __name__, current_line_number(),
                        f"Epsilon Greedy mixed {num_explore_schedules} random schedules into tuning set")
        else:
            self.logger(logging.INFO, __name__, current_line_number(),
                        f"Epsilon Greedy mixed {len(mixed_list)} random schedules into runner set")
        return mixed_list

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        # Check if there are any trials left
        if (self.st >= self.max_trials):
            return None

        # Check if next batch would go above max trial limit and adjust down
        sample_num = self.num_trials_per_iter
        if (self.ed > self.max_trials):
            sample_num = self.max_trials - self.st
            self.ed = self.max_trials

        assert self.st < self.ed, f"check failed: {self.st} < {self.ed}"

        # Top measured database schedules
        num_measured_schedules = int(self.search_strategy.population_size * self.search_strategy.init_measured_ratio)
        measured_schedules: List[Schedule] = self._pick_best_from_database(num_measured_schedules)

        # Sample a new population of random schedules
        num_unmeasured_schedules = self.search_strategy.population_size - len(measured_schedules)
        unmeasured_schedules: List[Schedule] = self._sample_initial_population(num_unmeasured_schedules)

        # Check if minimum amount of schedules were sampled
        if (len(unmeasured_schedules) < self.search_strategy.init_min_unmeasured):
            raise ValueError  # Specify a better error here

        self.logger(logging.INFO, __name__, current_line_number(),
                    f"Prepared a population of {len(measured_schedules) + len(unmeasured_schedules)} " +
                    "schedules for tuning selection")

        # Pick the random and untuned schedules for running
        random_schedules = self.epsilon_greedy_mix(exploit_list=[],
                                                   explore_list=unmeasured_schedules,
                                                   epsilon=0.2,
                                                   num=sample_num,
                                                   for_tuning=False)

        # Pick a mix of measured schedules and unmeasured for tuning.
        # The number of schedules send to the tuner is decided by how many random
        # schedules were selected for direct measurement.
        tune_schedules = self.epsilon_greedy_mix(exploit_list=measured_schedules,
                                                 explore_list=unmeasured_schedules,
                                                 epsilon=0.2,
                                                 num=sample_num - len(random_schedules),
                                                 for_tuning=True)
        tuned_schedules: List[Schedule] = self._send_to_bayesian_tuner(tune_schedules)

        run_schedules = random_schedules + tuned_schedules
        assert len(run_schedules) == sample_num
        return assemble_candidates(run_schedules)

    def _send_to_bayesian_tuner(self, tune_schedules: List[Schedule]) -> List[Schedule]:
        num_sch_to_tuner = len(tune_schedules)
        self.logger(logging.INFO, __name__, current_line_number(),
                    f"Sending {num_sch_to_tuner} schedule(s) to bayesian optimization tuner")

        def f_proc_bay_opt_tune(id: int):
            bay_opt_tuner = BayOptTuner(sch=tune_schedules[id],
                                        state=self,
                                        save_optimizer=True,
                                        validate_schedules=True)
            return id, bay_opt_tuner.tune()

        with Profiler.timeit("BayOptSearch/Tuner/Tune"):
            with ThreadPoolExecutor(max_workers=1) as executor:
            #with ThreadPoolExecutor(max_workers=self.context.num_threads) as executor:
                # Submit all tasks to the executor
                futures = [executor.submit(f_proc_bay_opt_tune, id) for id in range(num_sch_to_tuner)]

                # Process future results as they complete
                for future in as_completed(futures):
                    id, result = future.result()
                    tune_schedules[id] = result

        self.logger(logging.INFO, __name__, current_line_number(), "Bayesian optimization tuner finished")
        return tune_schedules

    def _sample_initial_population(self, num_traces: int) -> List[Schedule]:
        with Profiler.timeit("BayOptSearch/GenerateCandidates/SamplePopulation"):
            postproc = ThreadedTraceApply(self.search_strategy.postprocs)
            output_schedules: List[Schedule] = []
            fail_count: int = 0
            while (fail_count < self.search_strategy.max_fail_count and
                   len(output_schedules) < self.search_strategy.init_min_unmeasured):

                results = [None] * num_traces

                def f_proc_unmeasured(thread_id: int, trace_id: int):
                    thread_id = thread_id % self.context.num_threads
                    data: PerThreadData = self.per_thread_data_[thread_id]
                    rand_state: np.int64 = data.rand_state
                    mod: IRModule = data.mod

                    assert results[trace_id] is None, f"results {trace_id} should be None"

                    design_space_index: int = sample_int(rand_state, 0, len(self.design_spaces))
                    trace: Trace = Trace(self.design_spaces[design_space_index].insts, {})
                    sch: Schedule = postproc.apply(mod=mod, trace=trace, rand_state=rand_state)
                    if (sch is not None):
                        results[trace_id] = sch

                for i in range(num_traces):
                    f_proc_unmeasured(i, i)

                found_new: bool = False
                for i in range(num_traces):
                    if (results[i] is not None):
                        found_new = True
                        output_schedules.append(results[i])
                fail_count += not found_new
                self.logger(logging.INFO, __name__, current_line_number(),
                            f"Sampled {len(output_schedules)} new random schedules")
                return output_schedules

    def get_top_k_schedules(self, schedules: List[Schedule], k: int) -> List[Schedule]:
        with Profiler.timeit("BayOptSearch/GenerateCandidates/GetTopKSchedules"):
            scores = predict_normalized_scores(schedules, self.context, self.cost_model)
            idx = np.argsort(scores)[-k:]

            top_schedules: List[Schedule] = []
            for index in idx:
                top_schedules.append(schedules[index])
            return top_schedules

    def notify_runner_results(self, measure_candidates: List[MeasureCandidate], results: List[RunnerResult]):
        self.st += len(results)
        self.ed += len(results)


@derived_object
class BayesianOptimizationSearch(PySearchStrategy):
    context: "TuneContext" = None
    state: State = None

    population_size = 512
    init_measured_ratio = 0.2
    init_min_unmeasured = 50
    max_fail_count = 50

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the search strategy with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initialization.
        """
        self.context: TuneContext = context
        self.postprocs = context.space_generator.postprocs
        self.rand_state = forkseed(context.rand_state)

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
        # ToDo add checks here
        assert design_spaces is not None, "Design spaces should not be None!"
        if self.state is not None:
            print("ValueError: `PreTuning` is already invoked without corresponding `PostTuning`.")
            raise ValueError

        self.state = State(max_trials=max_trials,
                           num_trials_per_iter=num_trials_per_iter,
                           design_spaces_schedules=design_spaces,
                           database=database,
                           cost_model=cost_model,
                           search_strategy=self,
                           context=self.context)

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
