from itertools import chain
from typing import Iterable

from data import FeatVec
from hmm.state import State
from hmm.token_class import Token


def _move_tokens(tokens: Iterable[Token], observation: FeatVec) -> Iterable[Token]:
    return chain.from_iterable((t.forward(observation) for t in tokens))


def _filter_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
    tokens = sorted(tokens, key=lambda t: (id(t.state), t.log_probability), reverse=True)
    last_seen = None
    for token in tokens:
        if last_seen != id(token.state):
            last_seen = id(token.state)
            yield token


def viterbi(initial_state: State, observations: Iterable[FeatVec], target_state: State) -> Token:
    assert not initial_state.is_emitting
    active_tokens = [Token(initial_state)]
    for o in observations:
        active_tokens = list(_filter_tokens(_move_tokens(active_tokens, o)))
    active_tokens = sorted(_move_tokens(active_tokens, None), key=lambda t: t.log_probability, reverse=True)
    return next(filter(lambda t: id(t.state) == id(target_state), active_tokens))
