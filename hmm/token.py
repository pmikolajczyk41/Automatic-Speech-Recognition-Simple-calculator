from math import log
from typing import List

from data import FeatVec
from hmm.state import State


class Token:
    def __init__(self, state: State):
        assert not state.is_emitting
        self._state = state
        self.state_id = state.name
        self.history = (state.name,)
        self.log_probability = 0.

    def _new_token(self, new_state: State, transition_probability: float) -> 'Token':
        newborn = Token(new_state)
        newborn.history = self.history + newborn.history
        newborn.log_probability = self.log_probability + log(transition_probability)
        return newborn

    def forward(self, observation: FeatVec) -> List['Token']:
        newborns = []
        for new_state, transition_probability in zip(self._state.neigh, self._state.trans):
            new_token = self._new_token(new_state, transition_probability)
            if new_state.is_emitting:
                new_token.log_probability += new_state.emitting_logprobability(observation)
                newborns.append(new_token)
            else:
                newborns += new_token.forward(observation)
        return newborns
