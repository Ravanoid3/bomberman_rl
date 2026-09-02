"""The dihedral group D4 acting on features and actions.
The idea here is to easily gather extra data, by exploiting the symmetry of the Bomberman arena and the feature encoding.

The Bomberman arena is symmetric under the eight symmetries of the square, simliiarly to the feature encoding.

Every direction-dependent quantity is stored in a four-slot block ordered ``UP, RIGHT, DOWN, LEFT``.

A board symmetry therefore acts on a feature vector as a plain permutation of indices, and on an action as a
relabelling of the four movement actions.

We use this for 8x data augmentation during training.  A transition
``(phi, a, r, phi')`` is equally valid as ``(g.phi, g.a, r, g.phi')`` for every
``g`` in the group, which multiplies the effective sample size without any extra
environment interaction and pushes the learned Q-function towards the
equivariance the true Q-function has.
"""

import numpy as np

#: Quarter turn: UP -> RIGHT -> DOWN -> LEFT -> UP.
ROT = (1, 2, 3, 0)
#: Reflection about the vertical axis: swaps RIGHT and LEFT.
MIRROR = (0, 3, 2, 1)


def _compose(g, h):
    """Function composition ``(g o h)(d) = g[h[d]]``."""
    return tuple(g[h[d]] for d in range(4))


def _generate():
    identity = (0, 1, 2, 3)
    rotations = [identity]
    for _ in range(3):
        rotations.append(_compose(ROT, rotations[-1]))
    group = []
    for r in rotations:
        group.append(r)
        group.append(_compose(r, MIRROR))
    return tuple(group)


#: The eight elements of D4 as maps ``d -> g(d)`` on direction indices.
DIR_MAPS = _generate()
N_SYMMETRIES = len(DIR_MAPS)


def action_permutations(n_actions=6):
    """``perm[g, a]`` = the action ``a`` becomes under symmetry ``g``.
    Movement actions are relabelled, whilst ``WAIT`` and ``BOMB`` are fixed.
    """
    perms = np.zeros((N_SYMMETRIES, n_actions), dtype=np.int64)
    for gi, gmap in enumerate(DIR_MAPS):
        for a in range(n_actions):
            perms[gi, a] = gmap[a] if a < 4 else a
    return perms


def index_permutations(layout, size):
    """``perm[g]`` gathers a transformed feature vector: ``new = old[perm[g]]``.

    ``layout`` is an iterable of ``(name, base, width, is_direction_block)`` as
    produced by FeatureSpec in features.py.  For a direction block we want
    ``new[base + g(d)] == old[base + d]``, which as a gather reads
    ``new[base + j] == old[base + g^-1(j)]``.
    """
    perms = np.zeros((N_SYMMETRIES, size), dtype=np.int64)
    for gi, gmap in enumerate(DIR_MAPS):
        ginv = [0] * 4
        for d, image in enumerate(gmap):
            ginv[image] = d
        idx = np.arange(size)
        for _name, base, width, is_dir in layout:
            if is_dir:
                for j in range(4):
                    idx[base + j] = base + ginv[j]
        perms[gi] = idx
    return perms
