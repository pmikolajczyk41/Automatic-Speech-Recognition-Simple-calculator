from itertools import product
from math import log
from typing import List, Tuple

import numpy as np
from scipy.special import logsumexp

from data import FeatVec
from hmm.state import State, NULL_OBSERVATION

TP_EPS = 1e-15


def _compute_adjacency(states: List[State]) -> Tuple[List[List], List[List]]:
    e_in, e_out = [[] for _ in states], [[] for _ in states]
    for s in states:
        for n, tp in zip(s.neigh, s.trans):
            e_out[s.name].append((n.name, tp))
            e_in[n.name].append((s.name, tp))
    return e_in, e_out


def _compute_forward_backward(e_in, e_out, states, observations):
    n, m = len(states), len(observations)

    F, B = np.full((n, m), -np.inf), np.full((n, m), -np.inf)
    F[1, 0] = states[1].emitting_logprobability(observations[0])
    B[n - 1, -1] = 0.

    for j, i in product(range(1, m), range(1, n)):
        summands = [F[k, j - 1] + log(tp + TP_EPS) for k, tp in e_in[i]]
        F[i, j] = logsumexp(summands) + states[i].emitting_logprobability(observations[j])

    for j, i in product(range(m - 2, -1, -1), range(n - 2, 0, -1)):
        summands = [B[k, j + 1] + log(tp + TP_EPS) + states[k].emitting_logprobability(observations[j + 1])
                    for k, tp in e_out[i]]
        B[i, j] = logsumexp(summands)

    return F, B


def _compute_gamma(F, B):
    gamma = F + B
    denominator = logsumexp(gamma, axis=0)
    denominator[denominator == -np.inf] = 0.
    return gamma - denominator


def _compute_ksi(F, B, e_out, states, observations):
    n, m = F.shape
    ksi = np.full((n, n, m), -np.inf)
    for i, t in product(range(n), range(m - 1)):
        for j, tp in e_out[i]:
            ksi[i, j, t] = F[i, t] + B[j, t + 1] + log(tp + TP_EPS) \
                           + states[j].emitting_logprobability(observations[t + 1])

    denominator = logsumexp(ksi, axis=(0, 1))
    denominator[denominator == -np.inf] = 0.
    return ksi - denominator


def baum_welch(states: List[State], observations: List[FeatVec]):
    assert (not states[0].is_emitting) and (not states[-1].is_emitting)
    e_in, e_out = _compute_adjacency(states)
    observations = np.vstack([observations, NULL_OBSERVATION])
    F, B = _compute_forward_backward(e_in, e_out, states, observations)

    gamma = _compute_gamma(F, B)[:, :-1]
    ksi = _compute_ksi(F, B, e_out, states, observations)[:, :, :-1]
    gamma, ksi = np.nan_to_num(np.exp(gamma)), np.nan_to_num(np.exp(ksi))
    return gamma, ksi
