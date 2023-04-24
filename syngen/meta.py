import os
from numpy.random import Generator

from string import ascii_lowercase
from .types import Event


class BaseMetadataGenerator:
    def reset(self, origin: float = None) -> None:
        """Reset the metadata generator."""
        pass

    def step(self, random: Generator, event: Event) -> dict:
        """Process the event with the metadata generator.

        Notes
        -----
        Aside from adding meta info to the events, generated bya a markov chain
        core, the metadata generator also acts as an event filter: if it returns
        `None` as the result, then the event is to be skipped.
        """
        if event is None:
            return None

        t, label = event

        # an event name is everything before the first dunder in the state id
        # XXX `event__other__parts__of_id` -->> `event`
        label, _, _ = label.partition("__")
        if not label:
            return None

        # fallback to default `step` logic if there is no appropriate handler,
        #  of the event's label is has dunder prefix.
        handler = getattr(self, "step__" + label, lambda r, x: x)

        # replace the label of the event
        return handler(random, dict(event=label, ts=int(t)))

    def step__not_implemented(self, random: Generator, item: dict) -> dict:
        """the `step__*` methods specify the event post-processing logic."""
        raise NotImplementedError


class SimpleMetadataGenerator(BaseMetadataGenerator):
    def __init__(self, n_items: int) -> None:
        self.n_items = n_items

    def step__item_view(self, random: Generator, item: dict) -> dict:
        url, viewed_ = self.runtime

        # sample an id with replacement (repeated viewings)
        item_id = random.choice(self.n_items)

        # track the set of viewed items
        self.runtime = url, frozenset((*viewed_, item_id))  # XXX we make a copy

        # return the updated runtime state the metadata enhanced event
        return dict(**item, meta=dict(id=str(item_id)))

    def step__item_add(self, random: Generator, item: dict) -> dict:
        url, viewed_ = self.runtime

        # indicate invalid add event if nothing has been viewed so far
        if not viewed_:
            return None

        # `set.pop` returns an arbitrary element, but we shuffle anyway
        viewed = list(viewed_)
        random.shuffle(viewed)

        # drop an item
        item_id = viewed.pop()
        self.runtime = url, frozenset(viewed)

        # return metadata and the updated runtime
        return dict(**item, meta=dict(id=str(item_id)))

    def step__session_start(self, random: Generator, item: dict) -> dict:
        # sample a random string of lowercase letters
        n_letters = 5 + random.integers(11)  # XXX length [5..15]
        example = "".join(random.choice(list(ascii_lowercase), n_letters))

        self.runtime = f"http://{example}.com/", frozenset()
        return item

    def step__session_end(self, random: Generator, item: dict) -> dict:
        url, _ = self.runtime
        return item

    def step__page_view(self, random: Generator, item: dict) -> dict:
        url, viewed_ = self.runtime

        page_id = int(1 + random.integers(100))  # XXX id [1..100]
        duration = int(1 + random.integers(120))  # XXX seconds [1..120]

        meta = dict(url=os.path.join(url, f"page{page_id}"), duration=duration)
        return dict(**item, meta=meta)