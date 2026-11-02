import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from weaksense import label_model as lm, lfs, models, data


def _synthetic_votes(n=3000, seed=0):
    rng = np.random.default_rng(seed); y = (rng.random(n) < 0.3).astype(int)
    accs = [0.9, 0.85, 0.6, 0.55]; L = np.full((n, 4), -1)
    for j, a in enumerate(accs):
        act = rng.random(n) < 0.7; correct = rng.random(n) < a
        L[act, j] = np.where(correct[act], y[act], 1 - y[act])
    return L, y


def test_majority_vote_range():
    L, _ = _synthetic_votes(); p = lm.majority_vote(L); assert p.min() >= 0 and p.max() <= 1


def test_dawid_skene_learns_lf_ranking():
    L, y = _synthetic_votes(); ds = lm.DawidSkene().fit(L); acc = ds.learned_accuracy()
    assert acc[0] > acc[3] and acc[1] > acc[2]


def test_dawid_skene_beats_or_matches_mv():
    from sklearn.metrics import roc_auc_score
    L, y = _synthetic_votes(); assert roc_auc_score(y, lm.DawidSkene().fit(L).q) >= roc_auc_score(y, lm.majority_vote(L)) - 0.01


def test_anchors_are_respected():
    L, y = _synthetic_votes(); ds = lm.DawidSkene().fit(L, {0: 1.0, 1: 0.0}); assert ds.q[0] == 1 and ds.q[1] == 0


def test_lfs_shapes_and_values():
    rng = np.random.default_rng(0); W = rng.normal(size=(50, 32, 4)).astype(np.float32); W[:5, -1, 0] = 20
    xsm = lfs.CrossSensorModel().fit(rng.normal(size=(200, 4)))
    V, names = lfs.generic_lfs(W, xsm, per_sensor=True)
    assert V.shape == (50, len(names)) and set(np.unique(V)) <= {-1, 0, 1}
    assert (V[:5, 0] == 1).all()


def test_windows_no_future_leak():
    x = np.arange(200, dtype=float)[:, None].repeat(2, 1); y = np.zeros(200, int); y[150:] = 1
    W, yw, _ = data.windows(data.Run("t", x, y), 10, 50, stride=1)
    assert len(W) == 191 and yw[0] == y[9] and yw[140] == 0 and yw[141] == 1


def test_encoder_and_self_training_run():
    rng = np.random.default_rng(0); X = rng.normal(size=(300, 16, 3)).astype(np.float32); y = (rng.random(300) < .3)
    X[y] += 2; q = np.clip(y + rng.normal(0, .3, 300), 0, 1)
    m = models.self_train(X, q, np.arange(5), y[:5].astype(float), 0, rounds=1, epochs=3)
    p = models.predict(m, X); assert p.shape == (300,) and np.isfinite(p).all()


def test_nnpu_runs():
    rng = np.random.default_rng(0); X = rng.normal(size=(300, 16, 3)).astype(np.float32); pos = np.zeros(300, bool); pos[:20] = True
    p = models.predict(models.train_nnpu(X, pos, 0.3, 0, epochs=2), X); assert np.isfinite(p).all()
