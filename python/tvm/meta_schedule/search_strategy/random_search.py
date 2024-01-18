from typing import TYPE_CHECKING, List, Optional
from tvm.tir.schedule import Schedule, Trace

from .search_strategy import PySearchStrategy, MeasureCandidate, SearchStrategy
from ..utils import derived_object

if TYPE_CHECKING:
    from ..cost_model import CostModel
    from ..database import Database
    from ..tune_context import TuneContext


class State():
    max_trials: int = None
    num_trials_per_iter: int = None
    design_spaces: List[Schedule] = None
    database: Optional["Database"] = None
    cost_model: Optional["CostModel"] = None

    max_fail_count = 300

    def __init__(self,
                 max_trials: int = None,
                 num_trials_per_iter: int = None,
                 design_spaces: List[Schedule] = None,
                 database: Optional["Database"] = None,
                 cost_model: Optional["CostModel"] = None
                 ):
        self.max_trials = max_trials
        self.num_trials_per_iter = num_trials_per_iter
        self.design_spaces = design_spaces
        self.database = database
        self.cost_model = cost_model

    def reset(self):
        self.max_trials = None
        self.num_trials_per_iter = None
        self.design_spaces = None
        self.database = None
        self.cost_model = None

    def generate_measure_candidates(self) -> Optional[List[MeasureCandidate]]:
        schedules: List[MeasureCandidate] = _SampleInitialPopulation(self.state, self.min_sample, self.context.mod)

        return schedules


def _SampleInitialPopulation(state, min_sample, mod):
    schedules: List[MeasureCandidate] = []

    fail_count: int = 0
    while (fail_count < state.max_fail_count and len(schedules) < min_sample):

        seed = None

        trace: Trace = Trace()  # fill range

        sch: Schedule = Schedule(mod=mod, seed=seed, enable_check=True)
        trace.apply_to_schedule(sch=sch, remove_postproc=True)

        sch.enter_postproc()

        schedules.append(sch)


@derived_object
class RandomSearch(PySearchStrategy):
    min_sample: int = 50
    context: "TuneContext" = None
    state: State = None

    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the search strategy with tuning context.

        Parameters
        ----------
        context : TuneContext
            The tuning context for initialization.
        """
        self.context = context

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
        if self.state is not None:
            print("ValueError: `PreTuning` is already invoked without corresponding `PostTuning`.")
            raise ValueError

        self.state = State(max_trials=max_trials,
                           num_trials_per_iter=num_trials_per_iter,
                           design_spaces=design_spaces,
                           database=database,
                           cost_model=cost_model)

        print(self.state.database)

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

    def clone(self) -> SearchStrategy:
        """Clone the search strategy.

        Returns
        -------
        strategy : SearchStrategy
            The cloned search strategy.
        """
        return RandomSearch()
