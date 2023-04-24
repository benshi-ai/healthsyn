import numpy as np

from numpy import ndarray
from typing import Any

from scipy.special import softmax

from .types import Chain, ChainDict
from .types import Effect as Control, SimpleEffect as SimpleControl
from . import tcmc


class BaseBehavior:
    """Base class for markov chain behavior definitions."""

    def reset(self) -> int:
        raise NotImplementedError

    def apply(self, control: Any = None) -> tuple[ndarray, ndarray]:
        raise NotImplementedError


class MarkovBehavior(BaseBehavior):
    """Behavior for a markov chain, which is a mixture of one or more chains.

    Parameters
    ----------
    **chains: dict of Chain
        The named chains, from which to build the mixture. Each sub-dict specifies
        a markov chain through the average state holding times `hold`, state
        transitions probabilities `jump` and the initial state `init`.

    Attributes
    ----------
    weight: dict of float
        Nonzero behavior mixture weights.

    hold_: ndarray, shape = (n_chains, n_states,)
        The vector of average state holding times.

    jump_: ndarray, shape = (n_chains, n_states, n_states)
        The column-stochastic transition probability matrix.

    init_: int
        The initial state taken from the chain, which was specified __FIRST__
        during instantiation.

    labels_: list of str
        List of states defined by all behaviors.

    Notes
    -----
    The behavior object is fully initialized only upon calling `.reset`. During
    `__init__` we keep the chain specification in the lazy dictionary form.

    See Also
    --------
    tcmc.from_dict: Extract chain parameters from a dict repr.
    """

    def __init__(self, **chains: dict[str, Chain]) -> None:
        if not chains:
            raise ValueError("Must provide at least one markov chain definition.")

        # from archetype name to its index in the aligned hold-jump cube
        # XXX fix the chain enumeration
        self.names, self.chains = list(chains.keys()), chains

    def reset(self) -> int:
        """Reset the behavior to the default state.

        Returns
        -------
        init: int
            The state index form which the chain is to be started from.
        """

        # make sure the behaviors are aligned
        self.aligned_ = tcmc.from_many_chain_dicts(*map(self.chains.get, self.names))

        # reset to the default behavior (the first specified chain)
        # XXX unspecified values default to zero!
        self.weight = {self.names[0]: 1.0}

        # communicate the index of the starting state back top whoever asks
        # XXX the initial state is the one defined in the default behavior
        self.init_ = int(self.aligned_.init[0])
        return self.init_

    def to_label(self, jx: int) -> str:
        """Get the label of the specified state."""
        return self.aligned_.labels[jx]

    def to_state(self, label: str) -> int:
        """Get the state form the specified label."""
        return self.aligned_.labels.index(label)

    def __getstate__(self) -> dict:
        """Return a reduced state dict for more compact serialization."""

        state = self.__dict__.copy()  # XXX use `super().__getstate__` in py>=3.11

        if "aligned_" in state:
            del state["aligned_"], state["init_"]

        state["chains"] = {k: dict(c) for k, c in state["chains"].items()}
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state from a compact dict."""
        state["chains"] = {k: ChainDict(**c) for k, c in state["chains"].items()}

        # we reset then restore mixture weights and init
        # XXX use `super().__setstate__` in py>=3.11
        self.__dict__.update(state)

        self.reset()
        self.weight = dict(state["weight"])

    def apply(self, control: Control = None) -> tuple[ndarray, ndarray]:
        """Adjust the behavior in response to the control signal respecting
        the KL-div geometry of the probability simplex.

        Parameters
        ----------
        control: dict of float, default=None
            The control signal submitted to this behavioral core. The value at
            each key specifies a score, which reflects how much more likely the
            corresponding behavior should become.

        Notes
        -----
        We expect that the behavior NOT to change whenever control is None.
        """

        # get the ndarray of weights from a sparse dict
        w = np.array([self.weight.get(k, 0.0) for k in self.names], float)

        # apply control if supplied
        if control is not None:
            # control is a dict of floats keyed by the name of the behavior
            g = np.array([control.get(k, 0.0) for k in self.names], float)

            # we clip form below to avoid getting stuck at the boundary
            with np.errstate(divide="ignore"):  # XXX allow silent log(0) = -inf
                log_w = np.log(w).clip(-20)

            # updating the mixture in the specified direction staying within
            #  the probability simplex is done through a KL-div-regularized
            #  step in the direction `g` from the current `q`
            #    p \in \arg\min_{p \in \Delta_m} -g^\top p + KL(p\| q)
            w = softmax(log_w + g, axis=-1)

            # save only non-zero weights
            self.weight = {k: w[j] for j, k in enumerate(self.names) if w[j] > 0}

        # we make sure to always have a recomputed mixture
        return tcmc.blend(w, self.aligned_.hold, self.aligned_.jump, conditional=True)


class HoldingTimeScaler(MarkovBehavior):
    def __init__(self, base: MarkovBehavior, scale: float = 1.0) -> None:
        self.base, self.scale = base, scale

    def __getattr__(self, name: str) -> Any:
        # handle our own attrs, but delegate the rest to .base
        if name in ("base", "scale"):
            return getattr(self, name)

        base = getattr(self, "base")
        return getattr(base, name)

    def reset(self) -> int:
        return self.base.reset()

    def apply(self, control: Control = None) -> tuple[ndarray, ndarray]:
        hold, jump = self.base.apply(control)
        return hold / self.scale, jump


class TiredActive(MarkovBehavior):
    """Controlled linear interpolation between two behavioral archetypes.

    Parameters
    ----------
    tired: dict
        The markov chain corresponding to low activity behavior extreme.

    active: dict
        The markov chain corresponding to highly active behavior.

    Attributes
    ----------
    weight: dict of float
        Nonzero behavior mixture weights.

    hold_: ndarray, shape = (2, n_states,)
        The vector of average state holding times: tired 0, active 1.

    jump_: ndarray, shape = (2, n_states, n_states)
        The column-stochastic transition probability matrix: tired 0, active 1.

    init_: int
        The initial state taken from the chain, which was specified __FIRST__
        during instantiation.

    labels_: list of str
        List of states defined by both behaviors.
    """

    def __init__(self, tired: Chain, active: Chain):
        super().__init__(tired=tired, active=active)

    def apply(self, control: SimpleControl = None) -> tuple[ndarray, ndarray]:
        # `control` is the [0, 1] weight of active behavior
        if control is not None:
            # clip the weight to 0-1 range just to be safe
            theta = min(max(control, 0.0), 1.0)
            self.weight = dict(tired=1 - theta, active=theta)

        w = np.array([self.weight.get(k, 0.0) for k in self.names], float)
        return tcmc.blend(w, self.aligned_.hold, self.aligned_.jump, conditional=True)
