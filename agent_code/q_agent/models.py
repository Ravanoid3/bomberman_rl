"""Q-function approximators.

Two models behind one interface so that ``train.py`` is model-agnostic and the
tabular/linear comparison is a one-word config change:

``TabularQ``
    A dictionary from the raw feature bytes to a vector of action values.  No
    generalisation at all -- the honest baseline, and perfectly adequate once
    the curriculum has shrunk the state space (stage 1).

``LinearQ``
    ``Q(s, a) = w_a . phi(s)`` with per-action weight vectors, trained by
    semi-gradient Q-learning with RMSProp and a periodically synced target copy.
    Generalises across states that share features, which is what makes the
    ~10^5-state full game tractable.

Checkpoints record the feature layout they were trained with, so a warm start
into a *different* feature set transfers the blocks the two have in common and
zero-initialises the rest (see :func:`LinearQ.load_into`).
"""

import pickle
from pathlib import Path

import numpy as np

# WHY THE FUCK CAN PYTHON NOT HAVE AN INTERFACE MAN >:(
class QModel:
    """Common interface."""

    kind = 'abstract'

    def q_values(self, phi):
        raise NotImplementedError

    def q_batch(self, phis):
        raise NotImplementedError

    def q_batch_target(self, phis):
        return self.q_batch(phis)

    def update(self, phis, actions, targets):
        raise NotImplementedError

    def sync_target(self):
        pass

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as fh:
            pickle.dump(self.state_dict(), fh)

    def state_dict(self):
        raise NotImplementedError

    def stats(self):
        return {}


# Implementations
class TabularQ(QModel):
    kind = 'tabular'

    def __init__(self, n_actions, layout, lr=0.1, lr_decay=0.0, init=0.0):
        self.n_actions = n_actions
        self.layout = _layout_tuple(layout)
        self.lr = lr
        self.lr_decay = lr_decay
        self.init = init
        self.table = {}
        self.visits = {}

    def _row(self, key):
        row = self.table.get(key)
        if row is None:
            row = np.full(self.n_actions, self.init, dtype=np.float64)
            self.table[key] = row
            self.visits[key] = np.zeros(self.n_actions, dtype=np.int64)
        return row

    @staticmethod
    def key(phi):
        # Feature vectors travel through the pipeline as int8; casting here
        # keeps the dictionary keys stable no matter what the caller passes.
        return np.asarray(phi, dtype=np.int8).tobytes()

    def q_values(self, phi):
        row = self.table.get(self.key(phi))
        if row is None:
            return np.full(self.n_actions, self.init, dtype=np.float64)
        return row

    def q_batch(self, phis):
        return np.stack([self.q_values(p) for p in phis])

    def update(self, phis, actions, targets):
        errors = np.zeros(len(actions), dtype=np.float64)
        for i, (phi, a, target) in enumerate(zip(phis, actions, targets)):
            key = self.key(phi)
            row = self._row(key)
            visits = self.visits[key]
            visits[a] += 1
            alpha = self.lr / (1.0 + self.lr_decay * visits[a])
            err = target - row[a]
            row[a] += alpha * err
            errors[i] = err
        return errors

    def state_dict(self):
        return dict(kind=self.kind, n_actions=self.n_actions,
                    layout=self.layout, table=self.table, visits=self.visits,
                    lr=self.lr, lr_decay=self.lr_decay, init=self.init)

    def load_state(self, sd):
        if _layout_tuple(sd['layout']) != self.layout:
            return False  # incompatible representation: caller starts fresh
        self.table = sd['table']
        self.visits = sd['visits']
        return True

    def stats(self):
        return {'states': len(self.table)}


#
class LinearQ(QModel):
    kind = 'linear'

    def __init__(self, n_actions, layout, lr=0.02, grad_clip=5.0,
                 rmsprop_decay=0.95, eps=1e-6, init_scale=0.0):
        self.n_actions = n_actions
        self.layout = _layout_tuple(layout)
        self.n_features = sum(w for _n, _b, w in self.layout)
        self.lr = lr
        self.grad_clip = grad_clip
        self.rmsprop_decay = rmsprop_decay
        self.eps = eps

        rng = np.random.default_rng(0)
        self.W = (init_scale * rng.standard_normal((n_actions, self.n_features))
                  if init_scale else np.zeros((n_actions, self.n_features)))
        self.W_target = self.W.copy()
        self.ms = np.zeros_like(self.W)  # RMSProp running mean square

    def q_values(self, phi):
        return self.W @ np.asarray(phi, dtype=np.float64)

    def q_batch(self, phis):
        return np.asarray(phis, dtype=np.float64) @ self.W.T

    def q_batch_target(self, phis):
        return np.asarray(phis, dtype=np.float64) @ self.W_target.T

    def update(self, phis, actions, targets):
        phis = np.asarray(phis, dtype=np.float64)
        n_samples = len(actions)

        pred = (phis @ self.W.T)[np.arange(n_samples), actions]
        errors = np.clip(targets - pred, -self.grad_clip, self.grad_clip)

        # Per-action mean gradient without a Python loop: scatter the errors into
        # a one-hot (samples x actions) matrix and contract it with the features.
        # Profiling showed the naive per-action boolean-mask loop was the single
        # largest cost in training.
        scatter = np.zeros((n_samples, self.n_actions))
        scatter[np.arange(n_samples), actions] = errors
        grad = scatter.T @ phis                      # (n_actions, n_features)

        counts = np.bincount(actions, minlength=self.n_actions)
        touched = counts > 0
        grad[touched] /= counts[touched, None]

        self.ms[touched] = (self.rmsprop_decay * self.ms[touched]
                            + (1.0 - self.rmsprop_decay) * grad[touched] ** 2)
        self.W[touched] += (self.lr * grad[touched]
                            / (np.sqrt(self.ms[touched]) + self.eps))

        return errors

    def sync_target(self):
        self.W_target = self.W.copy()

    def state_dict(self):
        return dict(kind=self.kind, n_actions=self.n_actions,
                    layout=self.layout, W=self.W, lr=self.lr,
                    grad_clip=self.grad_clip)

    def load_state(self, sd):
        """Load weights, remapping feature blocks by name.

        Blocks present in both layouts are copied; blocks new to this model stay
        at zero.  This is what makes the curriculum warm starts work even though
        stage 2 drops the opponent blocks and stage 3 adds them back.
        """
        if sd.get('kind') != self.kind:
            return False
        src_layout = _layout_tuple(sd['layout'])
        src_W = sd['W']
        src_offsets = {name: (base, width) for name, base, width in src_layout}

        copied = 0
        for name, base, width in self.layout:
            if name in src_offsets:
                sbase, swidth = src_offsets[name]
                if swidth == width:
                    self.W[:, base:base + width] = src_W[:, sbase:sbase + swidth]
                    copied += width
        self.sync_target()
        return copied > 0

    def stats(self):
        return {'w_absmax': float(np.abs(self.W).max()),
                'w_norm': float(np.linalg.norm(self.W))}


# --------------------------------------------------------------------------
def _layout_tuple(layout):
    """Normalise a FeatureSpec layout to ``((name, base, width), ...)``."""
    out = []
    for entry in layout:
        name, base, width = entry[0], entry[1], entry[2]
        out.append((name, int(base), int(width)))
    return tuple(out)


def build(cfg, spec, n_actions):
    """Instantiate the model named by ``cfg['model']``."""
    if cfg['model'] == 'tabular':
        return TabularQ(n_actions, spec.layout, lr=cfg['lr'],
                        lr_decay=cfg['lr_decay'])
    if cfg['model'] == 'linear':
        return LinearQ(n_actions, spec.layout, lr=cfg['lr'],
                       grad_clip=cfg['grad_clip'])
    raise ValueError(f"unknown model {cfg['model']!r}")


def load_checkpoint(model, path, logger=None):
    """Try to restore ``model`` from ``path``; return True on success."""
    path = Path(path)
    if not path.is_file():
        if logger:
            logger.info(f'no checkpoint at {path}, starting from scratch')
        return False
    try:
        with open(path, 'rb') as fh:
            sd = pickle.load(fh)
            logger.info(f"Loaded checkpoint from {path}")
    except Exception as exc:  # pragma: no cover - corrupt file
        if logger:
            logger.warning(f'could not read {path}: {exc}')
        return False

    ok = model.load_state(sd)
    if logger:
        logger.info(f'checkpoint {path} -> {"loaded" if ok else "incompatible, starting fresh"}')
    return ok
