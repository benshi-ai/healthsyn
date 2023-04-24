import os
import tqdm

from sys import intern  # faster string comparison by interning
from time import sleep, strftime

from numpy.random import default_rng, Generator

import yaml
import json

from argparse import ArgumentParser, Namespace
from typing import Hashable, Callable, Any, Iterable
from typing_extensions import Self
from functools import partial
from itertools import count, chain


from .types import Event, NudgeMessage, SyngenDefn
from .behavior import MarkovBehavior, Control
from .sim import MarkovDynamics, BaseProcess, ProcessPool
from .meta import SimpleMetadataGenerator, BaseMetadataGenerator
from .utils import behavior_defn_from_yaml, to_effect_messages
from .utils import mkstemp, to_datetime

from .delayed import DelayedKeyboardInterrupt


def argparse() -> Namespace:
    parser = ArgumentParser(description="Start the algorithmic service.", add_help=True)
    parser.add_argument("path", type=str, help="Path where to dump json files.")
    parser.add_argument(
        "--timestamp",
        type=str,
        help="Provide start timestamp in RFC3339 format",
        default="2020-08-24T06:02:09Z",
    )
    parser.add_argument(
        "--definition",
        type=str,
        default="basic_behavior",
        help="Behavior and nudge reponse definition.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="syngen state checkpoint",
        default="",
    )
    args, _ = parser.parse_known_args()
    return args


# classes for log simulation
class Syndividual(BaseProcess):
    def __init__(self, core: BaseProcess, meta: BaseMetadataGenerator = None):
        super().__init__()
        self.core, self.meta = core, meta

    def reset(self, origin: float) -> Self:
        super().reset(origin)
        self.core.reset(origin)
        self.meta.reset(origin)
        return self

    def step(
        self, random, until: float, control: Control = None
    ) -> list[tuple[float, ...]]:
        assert self.started

        output = []

        # the core produces timestamp-label pairs
        for t, label in self.core.step(random, until, control):
            # .meta also acts as a filter
            item = self.meta.step(random, (t, label))
            if item is not None:
                output.append((t, item))

        # update internal clock
        super().step(random, until, None)
        assert self.time_ == self.core.time_
        return output


class SynPopulation(ProcessPool):
    def __init__(self, prefix: str = "user-") -> None:
        super().__init__()
        self.prefix, self._auto = prefix, count()

    def autoid(self):
        # We assign unique ids in a loop, since pool can be modified externally
        # XXX filter with map and str.format does not yaml dump-load very well
        while True:
            k = f"{self.prefix}{next(self._auto)}"
            if k not in self.pool:
                return k

    def add(self, p: Syndividual) -> Hashable:
        return super().add(self.autoid(), p)


def new_user(
    defn: SyngenDefn,
    n_items: int = 100,
) -> Syndividual:
    base = MarkovBehavior(**defn.archetypes)
    # base = HoldingTimeScaler(base, 10.0)
    core = MarkovDynamics(base)
    return Syndividual(core, SimpleMetadataGenerator(n_items))


def setup(
    definition: str = "basic_behavior",
    timestamp: str = "2020-08-24T06:02:09Z",
    n_users: int = 1000,
    n_items: int = 100,
) -> tuple[float, SynPopulation, dict, SyngenDefn]:
    from dateutil import parser

    # read the behaviour archetypes and nudge effects
    defn = behavior_defn_from_yaml(definition)

    # build the synthetic population
    # XXX if the consumer ever recevies an event from yet unseen individual id,
    #  then it creates a new user
    syn = SynPopulation()
    for _ in range(n_users):
        syn.add(new_user(defn, n_items))

    # `local` is in sync with the internal processes clocks
    local = parser.isoparse(timestamp).timestamp()

    # all users will have the same origin
    syn.reset(local)

    # partial session streams
    return local, syn, {}, defn


def is_session_end(payload: dict, label: str = intern("session_end")):
    return payload["event"] == label


def split(
    strands: dict,
    is_completed: Callable[Any, bool],
    syn: SynPopulation,
    random: Generator,
    until: float,
    control: list = None,
) -> dict[Hashable, list[Event[Any]]]:
    """Chunk the pooled user event stream into complete sessions."""
    chunks = {}  # XXX defaultdict(list)

    # fetch the next batch of events (they're ordered in ascending time)
    # XXX event's time, event's owner's id, event data
    for time, (synd, payload) in syn.ff(random, until, control):
        # add the next event to the partial session accumulator
        # XXX we save only the payload part
        strands.setdefault(synd, []).append(payload)

        if is_completed(payload):
            # commit the completed strand to the chunk storage
            chunks.setdefault(synd, []).append(strands.pop(synd))

    return chunks


def batchify(iterable: Iterable[Any], m: int) -> Iterable[tuple[Any]]:
    """Batch the iterable into tuples of length m."""
    batch = []
    for it in iterable:
        batch.append(it)
        if len(batch) > m:
            yield tuple(batch)
            batch = []

    if batch:
        yield tuple(batch)


def gather_nudges(folder: str, start: float, until: float) -> list[NudgeMessage]:
    """Monitor a folder with nudge JSONs and properly ingest them.

    Notes
    -----
    We assume ownership of any of the folder's content as soon as it is created.
    """
    return []  # (t, (u, nudge))


def save(
    filename: str, local: float, syn: SynPopulation, streams: dict, defn: SyngenDefn
) -> None:
    """Atomically make a YAML checkpoint."""

    # prepate the state and add some metadata to the checkpoint
    # XXX we use `.dict` to produce a simpler object for the definition
    ckpt = dict(
        __dttm__=strftime("%Y%m%d-%H%M%S"),
        time=local,
        population=syn,
        streams=streams,
        definition=defn.dict(),
    )

    try:
        tmp = mkstemp()
        with open(tmp, "wt") as f:
            yaml.dump(ckpt, f)

        os.replace(tmp, filename)

    except Exception:
        os.unlink(tmp)
        raise


def load(filename: str) -> tuple[float, SynPopulation, dict, SyngenDefn]:
    """Load state from a YAML."""

    with open(filename, "rt") as f:
        # get the state and rebuild the definition
        ckpt = yaml.load(f, Loader=yaml.CLoader)

    defn = SyngenDefn(**ckpt["definition"])
    return ckpt["time"], ckpt["population"], ckpt["streams"], defn


def main(
    path: str = "./logs",
    timestamp: str = "2020-08-24T06:02:09Z",
    definition: str = "basic_behavior",
    checkpoint: str = None,
    *,
    f_step: float = 60 * 60,
    f_sleep: float = 10.0,  # seconds to sleep after we caught up to dot-now
    f_share: float = 0.01,
    n_pop_cap: int = 10_000,  # 10k users max
    n_users_per_batch: int = 100,  # max 100 syndividual's per log
    f_checkpoint_period: float = 10,  # make a `latest` checkpoint every 10 sec
) -> None:
    from time import time as dot_now

    # default values for the initial pop size and the number of items in metadata
    n_users, n_items = 1000, 100

    # if the syngen state exists then restore local time, and partial streams
    if checkpoint is not None and os.path.isfile(checkpoint):
        local, syn, streams, defn = load(checkpoint)

    else:
        # init the local time, synthetic population and partial session streams
        local, syn, streams, defn = setup(
            definition, timestamp, n_users=n_users, n_items=n_items
        )

    # make sure the path is correct
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    # prepare the arena folder, whrein syngen check for nudges
    arena = None  # XXX hardcoded
    if arena is not None:
        arena = os.path.abspath(arena)

    # make sure we have a checkpoints subfolder ready
    checkpoints = os.path.join(path, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)

    # we always keep a backup of the `latest.yaml` checkpoint
    latest_ckpt = os.path.join(checkpoints, "latest.yaml")
    make_latest_ckpt = partial(save, latest_ckpt)
    make_latest_ckpt_bak = partial(os.replace, latest_ckpt, latest_ckpt + ".bak")

    # prepare the prng and flags
    random, caught_up = default_rng(None), False
    checkpoint_deadline = dot_now()
    with tqdm.tqdm(ncols=70, unit="s", unit_scale=True) as pb:
        try:
            n_max_arrivals_per_24h = None

            # catch up in leaps (o/w we may freeze for a while)
            # XXX syn does it internal timekeeping, the only requirement is that the
            #  its time origin is reasonably comparable to dot-now (although it can be
            #  arbitrary)
            do_spawn = partial(new_user, defn, n_items=n_items)
            do_ff = partial(split, streams, is_session_end)
            while True:
                with DelayedKeyboardInterrupt():
                    # checkpoint every once in a while before we do anything
                    if dot_now() > checkpoint_deadline:
                        pb.set_postfix_str("CHECKPOINT")
                        checkpoint_deadline = dot_now() + f_checkpoint_period

                        if os.path.isfile(latest_ckpt):
                            make_latest_ckpt_bak()
                        make_latest_ckpt(local, syn, streams, defn)

                    # leap f_step seconds ahead, and detect of we have caught up to
                    #  the `.now` wall time (UTC).
                    base, local = local, min(local + f_step, dot_now())
                    caught_up = (local >= dot_now()) or caught_up
                    pb.update(local - base)  # increment

                    # ingest the nudges and extract behaviorla effects from them
                    nudge_messages = gather_nudges(arena, base, local)
                    controls = to_effect_messages(defn, nudge_messages)

                    # fetch the next batch of complete sessions (`streams` is UPDATED)
                    # XXX same user's sessions are pasted together
                    sessions = {}
                    for u, chunks in do_ff(syn, random, local, controls).items():
                        sessions[u] = list(chain(*chunks))

                    # dump sessions if we got any and dump then into a JSON
                    # XXX we must design a way for an external observer to know if
                    #  an existing file is still being written to or is fully formed,
                    #  because io is slow. That is why we created a file under a
                    #  throwaway name in a `/tmp/...` folder, then sequentially fill
                    #  it with content, and only after that atomically move to
                    #  the destination. `os.replace` ensures atomicity.
                    # XXX https://docs.python.org/3/library/os.html#os.replace
                    if sessions:
                        # chunk the collected dict of sessions
                        batched = batchify(sessions.items(), n_users_per_batch)
                        for b, dict_chunk in enumerate(map(dict, batched)):
                            try:
                                # a unique filename somewhere in `/tmp/`
                                tmp = mkstemp()
                                with open(tmp, "wt") as f:
                                    json.dump(dict_chunk, f, indent=None)

                                logfile = os.path.join(
                                    path, f"session__{local:.0f}__{b:04d}.json"
                                )
                                os.replace(tmp, logfile)

                            except Exception:
                                os.unlink(tmp)

                                raise

                if caught_up:
                    sleep(f_sleep)

                # only update the per 24hrs rate when we CROSS a day boundary
                if (
                    n_max_arrivals_per_24h is None
                    or to_datetime(local).day != to_datetime(base).day
                ):
                    # adjust the per 24h arrival rate based relative to the pop size
                    n_max_arrivals_per_24h = random.integers(len(syn.pool) * f_share)

                # expand/shrink the population
                # XXX new users arrive randomly and independently of others
                #   into the population at a constant rate of µ per second
                #   (`f_arrivals_per_sec`). Thus `N(t, t+∆] ~ Pois(µ ∆)`
                # XXX the pop grows in discontinuous jumps, but linearly overall
                f_arrivals_per_sec = max(1, n_max_arrivals_per_24h) / (24 * 60 * 60)
                n_arrivals = min(
                    random.poisson(lam=(local - base) * f_arrivals_per_sec),
                    n_pop_cap - len(syn.pool),
                )
                n_departures = min(0, len(syn.pool))  # XXX hardcode to zero

                # remove some users forever -- they will no longer produce events
                if False:  # XXX DO NOT remove
                    # scan the pool for users that have recently entered a certain
                    #  imperfectly absorbing state, and randomly decide if they
                    #  a to be forced out
                    # XXX maintain a set of user-ids that have entered this state
                    #  at least once, and have survived the eviction
                    to_drop = random.choice(
                        list(syn.pool), size=n_departures, replace=False
                    )
                    for id in to_drop.tolist():
                        syn.remove(id)
                        if id in streams:
                            del streams[id]

                # add a random number of new users
                if len(syn.pool) < n_pop_cap:
                    for _ in range(n_arrivals):
                        syn.add(do_spawn().reset(local))

                pb.set_postfix_str(f"{len(syn.pool)} - {n_departures} + {n_arrivals}")

        except KeyboardInterrupt:
            print("Shutting down...")


main(**vars(argparse()))
