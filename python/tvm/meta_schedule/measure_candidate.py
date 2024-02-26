from typing import List

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule

from . import _ffi_api
from .arg_info import ArgInfo


@register_object("meta_schedule.MeasureCandidate")
class MeasureCandidate(Object):
    """Measure candidate class.

    Parameters
    ----------
    sch : tvm.tir.Schedule
        The schedule to be measured.
    args_info : List[ArgInfo]
        The argument information.
    """

    sch: Schedule
    args_info: List[ArgInfo]

    def __init__(
        self,
        sch: Schedule,
        args_info: List[ArgInfo],
    ) -> None:
        """Constructor.

        Parameters
        ----------
        sch : tvm.tir.Schedule
            The schedule to be measured.
        args_info : List[ArgInfo]
            The argument information.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureCandidate,  # type: ignore # pylint: disable=no-member
            sch,
            args_info,
        )