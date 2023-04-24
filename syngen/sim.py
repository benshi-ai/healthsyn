import numpy as np

from numpy.random import Generator

from functools import partial
from collections import defaultdict

from typing import Hashable, Callable, Any
from typing_extensions import Self

from . import tcmc
from .behavior import MarkovBehavior

from .types import Event


class BaseProcess:
    """A controllable continuous time stochastic process with arbitrary values.

    Attributes
    ----------
    started: bool
        Safety flag, that indicates if the processes' time and state were properly
        reset.

    origin_: float
        The origin of time.

    time_: float
        The current time of the process.
    """

    def __init__(self) -> None:
        self.started = False

    def reset(self, origin: float) -> Self:
        """Reset the processes' local time to the specified origin.

        Parameters
        ----------
        origin: float
            The local time, at which the process is to be started.

        Returns
        -------
        self
        """
        self.started = True

        # the local time starts at the specified origin
        self.origin_ = self.time_ = origin

        return self

    def step(
        self, random: Generator, until: float, control: Any = None
    ) -> list[Event[Any]]:
        """Simulate over the specified horizon having applied the impulse control.

        Parameters
        ----------
        random: Generator
            The source from which to draw randomness.

        until: float
            The local process time until which to run the simulation.

        control: any
            The control signal.

        Returns
        -------
        output: list of pairs
            The list of time-value pairs, representing the time at which
            a particular value has been emitted over the simulation interval.
        """

        assert self.started and self.time_ <= until and control is None

        self.time_ = until

        # an empty sequence of emitted signals for consistency
        return []

    def ff(
        self, random: Generator, until: float, controls: list[Event[Any]] = None
    ) -> list[Event[Any]]:
        """Fast forward until the specified time, while causally applying the controls.

        Parameters
        ----------
        random: Generator
            The source from which to draw randomness.

        until: float
            The local process time until which to fast-forward the simulation.

        control: list of pairs
            The list time-control pairs. Time is local to the processes' internal
            clock.

        Returns
        -------
        output: list of pairs
            The list of time-value pairs, representing the time at which
            a particular value has been emitted.

        Notes
        -----
        We reorder the control according to their times and split the simulation
        horizon into sub-intervals. If the processes' local time is `t_0`, the
        controls are `(t_k, u_k)_{k=1}^m`, and `until` equals `t_{m+1}`, then
        the simulation interval `(t_0, t_{m+1}]` is split into `(t_0, t_1]`,
        `(t_1, t_2]`, ... and `(t_m, t_∞]`. Sub-intervals are then simulated
        consecutively, taking into account that simulation over `(t_k, t_{k+1}]`
        MUST APPLY the impulse control `u_k`. The only exception is the zero-th
        interval `(t_0, t_1]` which is simulated without applying any control
        at all.

        The rationale is that the process evolves until `t_1` according to the
        previously established dynamics, while the control `u_k` INFLICTED at
        `t_k` AFFECTS the stochastic dynamics over the WHOLE FUTURE dynamics,
        especially over `(t_k, t_{k+1}]`. Thus we end up with the trajectory
        `(t_k, x_k)_{k ≥ 0}` where

            x_1 ~ p(X_{t_1} | X_{t_0}=x_0, U_{t_0}=ø),
            x_{k+1} ~ p(X_{t_{k+1}} | X_{t_k}=x_k, U_{t_k}=u_k),

        for all k = 1..m.
        """

        assert self.started
        output, evolve = [], partial(self.step, random)

        # advance the little-by-little until all piecewise controls have been applied
        s, u_s, T = self.time_, None, until
        assert s == self.time_
        for t, u_t in sorted(controls or [], key=lambda c: c[0]):
            if t < s:
                continue

            # evolve the process over (s, t] with x_t \sim p(.\mid x_s, u_s)
            output.extend(evolve(t, u_s))
            assert t == self.time_
            s, u_s = t, u_t

        if T >= s:
            output.extend(evolve(T, u_s))
            assert T == self.time_

        assert until == self.time_
        return output


class ProcessPool(BaseProcess):
    """A collection of independent controllable continuous time processes.

    Attributes
    ----------
    pool: dict
        The pool of processes in the group.
    """

    pool: dict[Hashable, BaseProcess]

    def __init__(self) -> None:
        super().__init__()
        self.pool = {}

    def reset(self, origin: float) -> Self:
        """Reset the pool to the origin.

        Parameters
        ----------
        origin: float
            The local time, at which the process is to be started.

        Returns
        -------
        self
        """
        super().reset(origin)
        for p in self.pool.values():
            p.reset(origin)

        return self

    def ff(
        self,
        random: Generator,
        until: float,
        controls: list[Event[[Hashable, Any]]] = None,
    ) -> list[Event[[Hashable, Any]]]:
        """Fast-forward through the processes, appropriately dispatching the controls.

        Parameters
        ----------
        random: Generator
            The source from which to draw randomness.

        until: float
            The local process time until which to fast-forward the simulation.

        control: list of pairs
            The list time-control pairs. The control also identifies, which process
            it is to be dispatched to. The corresponding times are local to each
            processes' internal clock.

            EXAMPLE: [(1.0, ("p1", CTRL))] indicates that the control CTRL is to
            be applied to process `p1` at its local time `t=1.0`.

        Returns
        -------
        output: list of pairs
            The list of time-value pairs, representing the time at which
            a particular value has been emitted.

        See Also
        --------
        BaseProcess.ff: fast-forwarding through a sequence of controls.
        """
        assert self.started

        # dispatch controls to each individual
        grouped = defaultdict(list)
        for t, (k, u_t) in controls or []:
            grouped[k].append((t, u_t))

        # simulate over (self.time_, until]
        events = []
        for k, u in self.pool.items():
            for t, x_t in u.ff(random, until, grouped[k]):
                events.append((t, (k, x_t)))

        super().step(random, until, None)
        return sorted(events, key=lambda it: it[0])

    def add(self, id: Hashable, p: BaseProcess) -> Hashable:
        """Add a running/started process to the pool."""
        if id in self.pool:
            raise KeyError

        self.pool[id] = p

        return id

    def remove(self, id: Hashable) -> None:
        """Remove a process from the pool."""
        del self.pool[id]


class MarkovDynamics(BaseProcess):
    """A continuous time markov chain controllable through its behavior core.

    Attributes
    ----------
    core: MarkovBehavior
        The behavior core, which affects the stochastic dynamics of the chain
        in response to controls.

    minimal_holding_time: float, default=1.0
        The minimal time between successive events.
    """

    def __init__(
        self, core: MarkovBehavior, *, minimal_holding_time: float = 1.0
    ) -> None:
        super().__init__()
        self.core, self.minimal_holding_time = core, minimal_holding_time

    def reset(self, origin: float) -> Self:
        """Reset the chain and behaviors.

        Parameters
        ----------
        origin: float
            The local time, at which the chain is to be started.

        Returns
        -------
        self
        """
        # call the base class to do the timekeeping
        super().reset(origin)

        # resetting the behavior core gives us also the initial state
        jx = int(self.core.reset())  # XXX avoid silent numpy scalars

        # reset and the local time the runtime state to a pre-history state
        self.runtime_ = [jx, -np.inf, origin]
        return self

    def step(
        self, random: Generator, until: float, control: Any = None
    ) -> list[Event[Any]]:
        """Simulate the chain after applying the impulse control at the current time.

        Parameters
        ----------
        random: Generator
            The source from which to draw randomness.

        until: float
            The local process time until which to simulate.

        control: any
            The control signal.

        Returns
        -------
        output: list of pairs
            The list of time-value pairs, representing the time at which
            a particular value has been emitted over the simulation interval.
        """
        assert self.started
        # we emit events as timestamp-payload pairs
        output = []

        # the chain emits the state labels only
        def emit(t, x):
            output.append((t, self.core.to_label(x[0])))

        # fast forward the chain until we end up inside a holding interval
        jx, t_0, t_1 = self.runtime_
        assert t_0 <= self.time_

        # update the dynamics based on the provided impulse control
        # XXX None-controls should not affect the stochastic dynamics
        hold_, jump_ = self.core.apply(control)

        # loop if [t_0, t_1) is in the past relative to `until`
        # XXX note that the hold and jump do not change since after the control
        #  took effect
        while t_1 <= until:  # XXX right-continuous piecewise constant
            # request the next state from the chain
            jx, t_0, t_1 = tcmc.step(random, (jx, t_0, t_1), (hold_, jump_))

            # ensure minimal holding time by modifying the chain's runtime state
            t_1 = max(t_1, self.minimal_holding_time + t_0)

            # emit the event
            emit(t_0, (jx, t_0, t_1))

        assert t_0 <= until < t_1

        # update the runtime
        self.runtime_ = [jx, t_0, t_1]

        # let the base class update the time
        super().step(random, until, None)
        return output


def clock(origin: float | str = None, speed: float = 1.0) -> Callable[None, float]:
    """Get the properly scaled live clock.

    Parameters
    ----------
    origin
    """
    # make sure clock speed is +ve
    if speed <= 0:
        raise ValueError(f"The clock speed should be +ve. Got {speed}.")

    # treat string as ISO 8601 timestamp (RFC 3339)
    if isinstance(origin, str):
        from time import monotonic  # monotonic float in seconds
        from dateutil import parser

        # posix timestamp (seconds since unix epoch)
        origin = parser.isoparse(origin).timestamp()

    elif origin is None:
        origin = monotonic()

    if not isinstance(origin, (int, float)):
        raise TypeError(f"origin must be an string or numeric. Got {type(origin)}.")

    # we localize `_base` in lambda for less scope lookups
    return lambda *, _base=monotonic(): origin + speed * (monotonic() - _base)
