import os
import yaml
import tempfile

from collections import defaultdict
from datetime import datetime


from . import tcmc
from .types import ChainDict, Effect, SyngenDefn, Nudge, NudgeMessage, EffectMessage


def mkstemp(suffix: str = None, prefix: str = None, dir: str = None) -> str:
    """Create and return a unique temporary file. The caller is responsible
    for deleting the file when done with it. Helpful for creating unquely
    named filenames anywhere.
    """

    fid, tempname = tempfile.mkstemp(suffix, prefix, dir, text=False)
    os.close(fid)

    return tempname


def resource_filename(filename: str, *, ext: str = ".yaml") -> str:
    """Check if the filename is absolute or if it is syngen's resource"""
    from pkg_resources import resource_filename

    if not os.path.isabs(filename):
        filename = resource_filename("syngen.data", filename + ext)

    if not os.path.isfile(filename):
        raise FileExistsError(f"Definition {filename} does not exist")

    return filename


def chain_from_yaml(chain: str) -> ChainDict:
    # verify the chain specification by loading it and converting to dict repr
    with open(resource_filename(chain, ext=".yaml"), "rt") as f:
        chain = tcmc.to_dict(tcmc.from_dict(**yaml.load(f, yaml.CLoader)))
        return ChainDict(**chain)


def behavior_defn(
    archetypes: dict[str, str], effects: dict[str, Effect] = None
) -> SyngenDefn:
    # sometimes effects may be missing, which means that noe are defined
    if effects is None:
        effects = {}

    # start by fetching the chains definitions and validating them
    archetypes_ = {name: chain_from_yaml(defn) for name, defn in archetypes.items()}

    # now process the specifications of nudge tag effects
    effects_ = {}
    for tag, effect in effects.items():
        # make sure the tag refers to predefined archetypes
        bad = effect.keys() - archetypes_.keys()
        if bad:
            raise ValueError(f"Undefined archetypes {bad} in tag {tag}.")

        bad = [n for n, v in effect.items() if not isinstance(v, (float, int))]
        if bad:
            raise TypeError(f"Non numeric values {bad} in {tag}.")

        # the effect vectors are shift invariant, so we make sure they're all +ve
        effects_[tag] = {t: v for t, v in effect.items() if v != 0}

    return SyngenDefn(archetypes=archetypes_, effects=effects_)


def behavior_defn_from_yaml(behavior: str) -> SyngenDefn:
    """Fetch the behaviors and nudge effect"""
    # fetch the definition
    with open(resource_filename(behavior, ext=".yaml"), "rt") as f:
        return behavior_defn(**yaml.load(f, yaml.CLoader))


def net_effect_from_tags(
    tags: list[str] | dict[str, float], defn: SyngenDefn
) -> Effect:
    """Compute the net effect of the nudge tags."""
    if isinstance(tags, list):
        tags = dict.fromkeys(tags, 1.0)

    if not isinstance(tags, dict):
        raise TypeError("nudge tags must be specified as a list of identifiers.")

    if not all(isinstance(v, float) for v in tags.values()):
        raise TypeError("tag weights must be numeric.")

    bad = [t for t in tags if t not in defn.effects]
    if bad:
        raise KeyError(f"Tags {bad} are missing from {set(defn.effects)}.")

    effect = defaultdict(float)  # XXX unspecified effects default to zero
    for tag in map(defn.effects.get, tags):
        for k, v in tag.items():
            effect[k] += float(v)

    # the effect vectors are shift invariant, so we make sure they're all +ve
    effect = {**dict.fromkeys(defn.archetypes, 0.0), **effect}
    base = float(min(effect.values()))
    return {t: v - base for t, v in effect.items() if v > base}


def to_effect_message(defn: SyngenDefn, message: NudgeMessage) -> EffectMessage:
    timestamp, (endpoint, nudge) = message
    assert isinstance(nudge, Nudge)
    return timestamp, (endpoint, net_effect_from_tags(nudge.tags, defn))


def to_effect_messages(
    defn: SyngenDefn, messages: list[NudgeMessage]
) -> list[EffectMessage]:
    return [to_effect_message(defn, m) for m in messages]


def to_datetime(timestamp: float) -> int:
    return datetime.fromtimestamp(timestamp)
