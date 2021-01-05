from itertools import chain
from typing import Iterable

from data import FeatVec
from hmm.state import State
from hmm.token_class import Token


def _move_tokens(tokens: Iterable[Token], observation: FeatVec) -> Iterable[Token]:
    new_tokens = (t.forward(observation) for t in tokens)
    return chain.from_iterable(new_tokens)


def _filter_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
    tokens = sorted(tokens, key=lambda t: (t.state_name, t.log_probability), reverse=True)
    last_seen = None
    for i, token in enumerate(tokens):
        if last_seen != token.state_name:
            last_seen = token.state_name
            yield token


def viterbi(initial_state: State, observations: Iterable[FeatVec], target_state: State) -> Token:
    assert not initial_state.is_emitting
    active_tokens = [Token(initial_state)]
    for o in observations:
        active_tokens = _filter_tokens(_move_tokens(active_tokens, o))
    active_tokens = _move_tokens(active_tokens, None)
    return next(filter(lambda t: t.state_name == target_state.name, active_tokens))
