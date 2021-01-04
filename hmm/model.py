from typing import List

from graphviz import Digraph

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

    def render(self) -> None:
        dot = Digraph(graph_attr={'rankdir': 'LR'}, node_attr={'shape': 'circle'})
        n = len(self._states)
        for s in self._states:
            if not s.is_emitting:
                dot.node(str(s.name), label='', width='0.2', style='filled')
        for u in range(n):
            for v, puv in self._adj[u]:
                dot.edge(str(u), str(v), label=str(puv))
        dot.render('model.gv', view=True)


if __name__ == '__main__':
    m = Model.Path(3)
    m.render()
