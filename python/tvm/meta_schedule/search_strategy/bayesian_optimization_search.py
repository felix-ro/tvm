from typing import TYPE_CHECKING, List, Optional, Any
from tvm.tir.schedule import Schedule, Trace, Instruction, InstructionKind
from tvm.ir import IRModule


from .search_strategy import PySearchStrategy, MeasureCandidate, SearchStrategy
from ..utils import derived_object
from ..arg_info import ArgInfo
from ..runner import RunnerResult

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database
    from ..tune_context import TuneContext

    ATTR_TYPE = Any

import numpy as np
import copy
import random


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


def predict_normalized_score(candidates, context, cost_model):
    """Predict the normalized score of a list of candidates."""
    assert len(candidates) != 0, "Candidates given for score prediction can not be empty list!"
    scores = cost_model.predict(context, assemble_candidates(candidates))

    # print(f"TLP predict = {scores}")
    scores = np.clip(scores, 0.0, np.inf)
    # print(f"TLP predict normal score = {scores}")
    return scores


class ThreadedTraceApply:
    class Item:
        postproc = None
        fail_counter = 0

        def __init__(self, postproc):
            self.postproc = postproc

    def __init__(self, postprocs) -> None:
        self.n_ = len(postprocs)
        self.items_ = [self.Item(postprocs[i]) for i in range(self.n_)]

    def apply(self, mod: IRModule, trace: Trace, rand_state: np.int64):
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
                 postprocs):
        self.sch: Schedule = sch
        self.postprocs = postprocs

    def tune(self):
        print("Tuning has started...")

        tune_inst: Instruction = None
        tune_attr: List[ATTR_TYPE]

        # Get a sample perfect tile instruction
        for inst in self.sch.trace.insts:
            kind: InstructionKind = inst.kind

            if (kind.name == "SamplePerfectTile"):
                tune_inst = inst
                tune_attr = inst.attrs

                print(tune_inst)
                print(tune_attr)

                print(self.sch.trace.show())

                trace = self.sch.trace.with_decision(tune_inst, [4, 64], True)
                print(trace.show())

                pp = ThreadedTraceApply(postprocs=self.postprocs)
                return pp.apply(self.sch.mod, trace, 1)

                # self.sch.trace.apply_to_schedule(sch=self.sch, remove_postproc=True)
                return self.sch


class State:
    def __init__(self,
                 max_trials: int,
                 num_trials_per_iter: int,
                 design_spaces_schedules: List[Schedule],
                 database: Optional["Database"],
                 cost_model: Optional["CostModel"],
                 search_strategy: "BayesianOptimizationSearch",
                 context
                 ):
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_space_schedules = design_spaces_schedules
        self.database = database
        self.cost_model = cost_model
        self.search_strategy: BayesianOptimizationSearch = search_strategy
        self.context = context
        self.mod = context.mod

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

    def reset(self):
        self.max_trials = None
        self.num_trials_per_iter = None
        self.design_spaces = None
        self.database = None
        self.cost_model = None

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

        # Sample a new population of random schedules
        unmeasured_schedules: List[Schedule] = self.sample_initial_population(self.search_strategy.population_size)

        # Check if minimum amount of schedules were sampled
        if (len(unmeasured_schedules) < self.search_strategy.init_min_unmeasured):
            raise ValueError  # Specify a better error here

        # Pick top-k best schedules using cost func to avoid measuring all schedules
        top_k_schedules = self.get_top_k_schedules(unmeasured_schedules, sample_num)

        # Testing tuning start
        bay_opt_tuner = BayOptTuner(top_k_schedules[0], self.search_strategy.postprocs)
        top_k_schedules[0] = bay_opt_tuner.tune()
        # Testing tuning end

        return assemble_candidates(top_k_schedules)

    def sample_initial_population(self, num_traces: int) -> List[Schedule]:
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
            return output_schedules

    def get_top_k_schedules(self, schedules: List[Schedule], k: int) -> List[Schedule]:
        scores = predict_normalized_score(schedules, self.context, self.cost_model)
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
        self.context = context
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
