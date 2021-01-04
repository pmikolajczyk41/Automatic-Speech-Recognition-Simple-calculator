from typing import List

from hmm.state import State, default_distribution


class Model:
    @classmethod
    def Path(cls, emitting_length: int):
        middle = [State(default_distribution, True) for _ in range(emitting_length)]
        states = [State()] + middle + [State()]

        for p, n in zip(states, states[1:]):
            p.add_neigh(n, 0.5)

        return Model(states)

    def __init__(self, states: List[State]):
        self._states = states
        for id, state in enumerate(states):
            state.name = id
        self._adj = [[(n.name, tp) for n, tp in zip(s.neigh, s.trans)]
                     for s in states]
