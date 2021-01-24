import sys
from itertools import chain
from typing import Iterable, List

from data import FeatVec
from hmm.state import State
from hmm.token_class import Token


def _move_tokens(tokens: Iterable[Token], observation: FeatVec) -> Iterable[Token]:
    return chain.from_iterable((t.forward(observation) for t in tokens))


def _filter_tokens(tokens: Iterable[Token], beam_size: int, margin_size: float) -> List[Token]:
    tokens = sorted(tokens, key=lambda t: (id(t.state), t.log_probability), reverse=True)
    last_seen = None
    filtered = []
    for token in tokens:
        if last_seen != id(token.state):
            last_seen = id(token.state)
            filtered.append(token)
    tokens = sorted(filtered, key=lambda t: t.log_probability, reverse=True)[:beam_size]
    threshold = tokens[0].log_probability - margin_size
    return list(filter(lambda t: t.log_probability >= threshold, tokens))


def viterbi(initial_state: State, observations: Iterable[FeatVec], target_state: State,
            beam_size: int = sys.maxsize, margin_size: float = sys.maxsize) -> Token:
    assert not initial_state.is_emitting
    active_tokens = [Token(initial_state)]
    for o in observations:
        active_tokens = _filter_tokens(_move_tokens(active_tokens, o), beam_size, margin_size)
    active_tokens = sorted(_move_tokens(active_tokens, None), key=lambda t: t.log_probability, reverse=True)
    return next(filter(lambda t: id(t.state) == id(target_state), active_tokens))
