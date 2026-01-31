
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Optional GPU backend (PyTorch). If CUDA is available on the user's machine,
# the evaluator can run on GPU without changing the pipeline logic.
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


# ---------------------------
# Reproducibility / time
# ---------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def now() -> float:
    return time.perf_counter()


# ---------------------------
# Rank/cosine (no scipy)
# ---------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j) + 1.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom == 0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ---------------------------
# Graph / weights
# ---------------------------

@dataclass
class WeightedGraph:
    n: int
    W: np.ndarray
    family: str
    seed: int

    # spins is not required for the pipeline; we keep the field for backwards
    # compatibility but do not store it by default (memory-heavy for n>=18).
    spins: np.ndarray = None
    ising_energy: np.ndarray = None
    cut_value: np.ndarray = None
    W_sum: float = 0.0

    def build_cache(self) -> None:
        # Vectorized construction of Z-basis spin configurations and energies.
        # This is crucial for n up to ~20 (2^n states).
        n = int(self.n)
        N = 1 << n

        # bit-matrix: shape (N,n), where column i is bit i (LSB is i=0).
        z = np.arange(N, dtype=np.uint32)[:, None]
        bits = (z >> np.arange(n, dtype=np.uint32)[None, :]) & 1
        # spins in {+1,-1} as float32 to reduce memory; energies are accumulated in float64.
        s = (1.0 - 2.0 * bits.astype(np.float32))

        self.W_sum = float(np.sum(np.triu(self.W, 1)))
        v = s @ self.W.astype(np.float32)
        ising = 0.5 * np.sum(v * s, axis=1, dtype=np.float64)
        self.ising_energy = ising.astype(np.float64, copy=False)
        self.cut_value = (0.5 * self.W_sum - 0.5 * self.ising_energy).astype(np.float64, copy=False)

        # Do not keep the full spins matrix unless explicitly needed.
        self.spins = None
        del z, bits, s, v


def er_dense_p05_weighted(n: int, seed: int, p: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < p
    mask = np.triu(mask, 1)
    w = rng.random((n, n))
    W = mask * w
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    return W


def rr_k_regular(n: int, k: int, seed: int, max_tries: int = 5000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assert (n * k) % 2 == 0
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), k)
        rng.shuffle(stubs)
        edges = stubs.reshape(-1, 2)
        if np.any(edges[:, 0] == edges[:, 1]):
            continue
        a = np.minimum(edges[:, 0], edges[:, 1])
        b = np.maximum(edges[:, 0], edges[:, 1])
        pairs_view = a * n + b
        if len(np.unique(pairs_view)) != len(pairs_view):
            continue
        W = np.zeros((n, n), dtype=float)
        W[a, b] = 1.0
        W[b, a] = 1.0
        return W
    raise RuntimeError("Failed to sample k-regular graph")


def complete_weighted_dense(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.random((n, n))
    W = np.triu(W, 1)
    W = W + W.T
    np.fill_diagonal(W, 0.0)
    return W


def build_graph(family: str, n: int, seed: int) -> WeightedGraph:
    if family == "ER_dense_p05_weighted":
        W = er_dense_p05_weighted(n, seed, p=0.5)
    elif family == "RR_3regular":
        W = rr_k_regular(n, 3, seed)
    elif family == "Complete_weighted_dense":
        W = complete_weighted_dense(n, seed)
    else:
        raise ValueError(f"Unknown family: {family}")
    g = WeightedGraph(n=n, W=W, family=family, seed=seed)
    g.build_cache()
    return g


# ---------------------------
# Basis / statevector
# ---------------------------

def all_spins(n: int) -> np.ndarray:
    """Return all Z-basis spin configurations s_i in {+1,-1}.

    Vectorized (no Python loops): feasible up to n≈20.
    Column i corresponds to bit i (LSB is i=0).
    """
    N = 1 << n
    z = np.arange(N, dtype=np.uint32)[:, None]
    bits = (z >> np.arange(n, dtype=np.uint32)[None, :]) & 1
    spins = (1.0 - 2.0 * bits.astype(np.float32)).astype(np.float32, copy=False)
    return spins


def plus_state(n: int) -> np.ndarray:
    N = 1 << n
    return np.ones(N, dtype=np.complex128) / math.sqrt(N)


def apply_cost_phase(psi: np.ndarray, gamma: float, ising_energy: np.ndarray) -> np.ndarray:
    return psi * np.exp(-1j * gamma * ising_energy)


def apply_rx_all(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply exp(-i beta sum_i X_i) as Rx(2*beta) on every qubit.

    IMPORTANT: uses copy() on slices to preserve unitarity.
    """
    c = math.cos(beta)
    s = -1j * math.sin(beta)

    out = psi.copy()
    for q in range(n):
        step = 1 << q
        period = step << 1
        for base in range(0, len(out), period):
            i0 = base
            i1 = base + step
            a = out[i0 : i0 + step].copy()
            b = out[i1 : i1 + step].copy()
            out[i0 : i0 + step] = c * a + s * b
            out[i1 : i1 + step] = s * a + c * b
    return out


# ---------------------------
# Optional Torch evaluator (CPU/GPU)
# ---------------------------

def _torch_require() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is not available. Install torch with CUDA support (or use --backend numpy)."
        )


def _torch_parse_device(device: str) -> "torch.device":
    _torch_require()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _torch_complex_dtype(name: str):
    _torch_require()
    if name == "complex128":
        return torch.complex128
    return torch.complex64


def plus_state_torch(n: int, device: "torch.device", dtype) -> "torch.Tensor":
    _torch_require()
    N = 1 << n
    return (torch.ones(N, device=device, dtype=dtype) / math.sqrt(N)).clone()


def apply_cost_phase_torch(psi: "torch.Tensor", gamma: float, ising_energy: "torch.Tensor") -> "torch.Tensor":
    # psi <- psi * exp(-i gamma E_z)
    return psi * torch.exp((-1j * float(gamma)) * ising_energy)


def apply_rx_all_torch(psi: "torch.Tensor", beta: float, n: int) -> "torch.Tensor":
    """Apply exp(-i beta sum_i X_i) as Rx(2*beta) on every qubit.

    Bit ordering matches the numpy implementation: qubit q corresponds to bit q in the basis index.
    """
    c = math.cos(float(beta))
    s = -1j * math.sin(float(beta))
    out = psi
    # In-place updates using views; clones prevent overwriting within a layer.
    for q in range(n):
        step = 1 << q
        period = step << 1
        blocks = out.view(-1, period)
        a = blocks[:, :step].clone()
        b = blocks[:, step:].clone()
        blocks[:, :step] = c * a + s * b
        blocks[:, step:] = s * a + c * b
    return out


class QAOAEvaluatorTorch:
    """Exact statevector QAOA evaluator using PyTorch (optionally CUDA).

    This keeps the *same* objective as the numpy evaluator: expected weighted MaxCut.
    """

    def __init__(
        self,
        graph: WeightedGraph,
        p: int,
        device: str = "auto",
        dtype: str = "complex64",
        real_dtype: str = "float32",
    ):
        _torch_require()
        self.g = graph
        self.p = int(p)
        self.stats = EvalStats()
        self.device = _torch_parse_device(device)
        self.dtype = _torch_complex_dtype(dtype)
        self.real_dtype = torch.float64 if (real_dtype == "float64") else torch.float32

        # Cache data on the chosen device.
        self._ising = torch.tensor(self.g.ising_energy, device=self.device, dtype=self.real_dtype)
        self._cut = torch.tensor(self.g.cut_value, device=self.device, dtype=self.real_dtype)
        self._psi0 = plus_state_torch(self.g.n, self.device, self.dtype)

    def reset_stats(self) -> None:
        self.stats = EvalStats()

    def expected_cut(self, theta: np.ndarray) -> float:
        t0 = now()
        self.stats.n_evals += 1

        with torch.no_grad():
            betas, gammas = split_theta(theta, self.p)
            psi = self._psi0.clone()
            for l in range(self.p):
                psi = apply_cost_phase_torch(psi, float(gammas[l]), self._ising)
                psi = apply_rx_all_torch(psi, float(betas[l]), self.g.n)

            probs = psi.abs().pow(2).to(dtype=self.real_dtype)
        ps = float(probs.sum().item())
        if not (0.999999 <= ps <= 1.000001):
            if ps <= 0 or abs(ps - 1.0) > 1e-3:
                raise RuntimeError(f"Statevector not normalized (sum probs={ps})")
            probs = probs / ps
        val = float((probs * self._cut).sum().item())
        self.stats.wall_s += now() - t0
        return val


def make_evaluator(graph: WeightedGraph, p: int, backend: str, device: str, dtype: str, real_dtype: str = "float32"):
    """Factory: return a QAOA evaluator for (graph,p) using numpy or torch."""
    if backend == "torch":
        return QAOAEvaluatorTorch(graph, p, device=device, dtype=dtype, real_dtype=real_dtype)
    return QAOAEvaluator(graph, p)


# ---------------------------
# Evaluator + stats
# ---------------------------

@dataclass
class EvalStats:
    n_evals: int = 0
    wall_s: float = 0.0


class QAOAEvaluator:
    def __init__(self, graph: WeightedGraph, p: int):
        self.g = graph
        self.p = p
        self.stats = EvalStats()
        # Cache |+>^n to avoid re-allocating it at every evaluation.
        self._psi0 = plus_state(self.g.n)

    def reset_stats(self) -> None:
        self.stats = EvalStats()

    def expected_cut(self, theta: np.ndarray) -> float:
        t0 = now()
        self.stats.n_evals += 1

        betas, gammas = split_theta(theta, self.p)
        psi = self._psi0.copy()
        for l in range(self.p):
            psi = apply_cost_phase(psi, gammas[l], self.g.ising_energy)
            psi = apply_rx_all(psi, betas[l], self.g.n)

        probs = np.abs(psi) ** 2
        ps = float(np.sum(probs))
        if not (0.999999 <= ps <= 1.000001):
            if ps <= 0 or abs(ps - 1.0) > 1e-3:
                raise RuntimeError(f"Statevector not normalized (sum probs={ps})")
            probs = probs / ps
        val = float(np.dot(probs, self.g.cut_value))

        self.stats.wall_s += now() - t0
        return val


def split_theta(theta: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    assert theta.shape[0] == 2 * p
    return theta[:p], theta[p:]


def clip_theta(theta: np.ndarray, p: int) -> np.ndarray:
    betas, gammas = split_theta(theta, p)
    betas = np.clip(betas, 0.0, 0.5 * math.pi)
    gammas = np.mod(gammas, math.pi)
    return np.concatenate([betas, gammas])


def select_best_scaled_init_eval_only(
    evaluator: "QAOAEvaluator",
    theta_in: np.ndarray,
    p: int,
    beta_mults: List[float],
    gamma_mults: List[float],
    base_val: Optional[float] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Evaluate a small (beta,gamma) scaling grid on the *target* (eval-only).

    Important: this function does *not* run any optimizer; it only evaluates
    candidate scaled initializations and picks the best.

    Returns (best_theta, best_val, best_beta_mult, best_gamma_mult).
    """
    if base_val is None:
        base_val = float(evaluator.expected_cut(theta_in))

    best_theta = np.asarray(theta_in, dtype=float)
    best_val = float(base_val)
    best_bm = 1.0
    best_gm = 1.0

    betas0, gammas0 = split_theta(best_theta, p)
    for bm in beta_mults:
        for gm in gamma_mults:
            if abs(bm - 1.0) < 1e-12 and abs(gm - 1.0) < 1e-12:
                continue
            betas = np.clip(betas0 * bm, 0.0, 0.5 * math.pi)
            gammas = np.clip(gammas0 * gm, 0.0, math.pi)
            th = np.concatenate([betas, gammas])
            v = float(evaluator.expected_cut(th))
            if v > best_val:
                best_val = v
                best_theta = th
                best_bm = float(bm)
                best_gm = float(gm)

    return best_theta, best_val, best_bm, best_gm


def random_theta(p: int, rng: np.random.Generator) -> np.ndarray:
    betas = rng.random(p) * (0.5 * math.pi)
    gammas = rng.random(p) * (math.pi)
    return np.concatenate([betas, gammas])




def lift_theta(theta: np.ndarray, p_from: int, p_to: int, mode: str = "repeat_last") -> np.ndarray:
    """Lift or truncate QAOA parameters from depth p_from to p_to.

    Needed when you match behavior fingerprints at depth p_match but run the
    target optimization at depth p_target.

    mode:
      - repeat_last: pad extra layers by repeating the last (beta,gamma)
      - repeat:      tile the whole sequence
      - pad_zero:    pad extra layers with zeros
    """
    theta = np.asarray(theta, dtype=float)
    assert theta.shape[0] == 2 * p_from

    if p_from == p_to:
        return theta.copy()

    betas, gammas = split_theta(theta, p_from)

    if p_to < p_from:
        betas2 = betas[:p_to]
        gammas2 = gammas[:p_to]
    else:
        extra = p_to - p_from
        if mode == "pad_zero":
            betas_pad = np.zeros(extra, dtype=float)
            gammas_pad = np.zeros(extra, dtype=float)
        elif mode == "repeat":
            betas_pad = np.tile(betas, int(np.ceil(extra / p_from)))[:extra]
            gammas_pad = np.tile(gammas, int(np.ceil(extra / p_from)))[:extra]
        else:  # repeat_last
            betas_pad = np.full(extra, float(betas[-1]), dtype=float)
            gammas_pad = np.full(extra, float(gammas[-1]), dtype=float)

        betas2 = np.concatenate([betas, betas_pad])
        gammas2 = np.concatenate([gammas, gammas_pad])

    out = np.concatenate([betas2, gammas2])
    return clip_theta(out, p_to)


def transfer_and_select_init_eval_only(
    eval_target: QAOAEvaluator,
    theta_src: np.ndarray,
    p_src: int,
    p_target: int,
    *,
    gamma_scale: float,
    lift_mode: str,
    use_2d_scaling_grid: bool,
    beta_mults: List[float],
    gamma_mults: List[float],
    warm_gamma_mults: List[float],
) -> Tuple[np.ndarray, float, float, float]:
    """Map a donor theta to the target and pick a best eval-only scaled init.

    Returns (best_init_theta, best_init_val, beta_mult_best, gamma_mult_best).

    Important: evaluation only (no target optimization).
    """
    betas_src, gammas_src = split_theta(theta_src, p_src)
    gammas_tr = np.clip(gammas_src * float(gamma_scale), 0.0, math.pi)
    theta_tr = np.concatenate([betas_src, gammas_tr])
    theta_tr = lift_theta(theta_tr, p_src, p_target, mode=lift_mode)

    tr_raw = float(eval_target.expected_cut(theta_tr))

    beta_mult_best = 1.0
    gamma_mult_best = 1.0

    if use_2d_scaling_grid:
        best_init, best_val, beta_mult_best, gamma_mult_best = select_best_scaled_init_eval_only(
            eval_target,
            theta_tr,
            p_target,
            beta_mults=beta_mults,
            gamma_mults=gamma_mults,
            base_val=tr_raw,
        )
        return best_init, float(best_val), float(beta_mult_best), float(gamma_mult_best)

    # 1D gamma-mult search (eval-only)
    best_init = np.asarray(theta_tr, dtype=float)
    best_val = float(tr_raw)
    b, g = split_theta(theta_tr, p_target)
    for mlt in warm_gamma_mults:
        if abs(float(mlt) - 1.0) < 1e-12:
            continue
        th = np.concatenate([b, np.clip(g * float(mlt), 0.0, math.pi)])
        v = float(eval_target.expected_cut(th))
        if v > best_val:
            best_val = v
            best_init = th
            gamma_mult_best = float(mlt)
    return best_init, float(best_val), float(beta_mult_best), float(gamma_mult_best)

# ---------------------------
# SPSA optimizer (maximize)
# ---------------------------

@dataclass
class OptTrace:
    best: float
    best_theta: np.ndarray
    history: List[float]


def spsa_optimize(
    evaluator: QAOAEvaluator,
    theta0: np.ndarray,
    iters: int,
    seed: int,
    a: float,
    c: float,
    alpha: float = 0.602,
    gamma: float = 0.101,
    A: float = 10.0,
    clip_norm: Optional[float] = None,
    update_mask: Optional[np.ndarray] = None,
) -> OptTrace:
    """SPSA for maximizing f(theta). Uses 2 evals per iteration."""
    rng = np.random.default_rng(seed)
    p = evaluator.p
    theta = clip_theta(theta0.copy(), p)

    best = -float("inf")
    best_theta = theta.copy()
    hist: List[float] = []

    def f(th: np.ndarray) -> float:
        return evaluator.expected_cut(clip_theta(th, p))

    for k in range(1, iters + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k ** gamma)

        # Rademacher perturbation
        delta = rng.choice([-1.0, 1.0], size=theta.shape[0]).astype(float)
        if update_mask is not None:
            # keep perturbations only on active coords to avoid wasting evals
            delta = delta * update_mask
            # if mask zeros everything, fallback to all-ones to avoid NaNs
            if np.all(delta == 0.0):
                delta = rng.choice([-1.0, 1.0], size=theta.shape[0]).astype(float)

        th_plus = theta + ck * delta
        th_minus = theta - ck * delta
        f_plus = f(th_plus)
        f_minus = f(th_minus)

        # gradient estimate
        ghat = (f_plus - f_minus) / (2.0 * ck) * delta
        if update_mask is not None:
            ghat = ghat * update_mask

        if clip_norm is not None:
            ng = float(np.linalg.norm(ghat))
            if ng > clip_norm and ng > 0:
                ghat = ghat * (clip_norm / ng)

        theta = theta + ak * ghat
        theta = clip_theta(theta, p)

        # track best using the best of plus/minus/current quickly
        f_cur = max(f_plus, f_minus)
        if f_cur > best:
            best = f_cur
            best_theta = clip_theta(th_plus if f_plus >= f_minus else th_minus, p)
        hist.append(best)

    # final eval at best_theta (1 eval) for reporting consistency
    f_best = evaluator.expected_cut(best_theta)
    if f_best > best:
        best = f_best
    hist.append(best)
    return OptTrace(best=float(best), best_theta=best_theta.copy(), history=hist)


def scheduled_spsa(
    evaluator: QAOAEvaluator,
    theta0: np.ndarray,
    iters: int,
    seed: int,
    a: float,
    c: float,
    alpha: float,
    gamma: float,
    A: float,
    ls_steps: int = 0,
    ls_layers: int = 1,
    clip_norm: Optional[float] = None,
) -> OptTrace:
    """Two-stage layer-selective SPSA: last layers first, then all."""
    if ls_steps <= 0 or ls_layers <= 0 or ls_steps >= iters:
        return spsa_optimize(
            evaluator, theta0, iters, seed, a=a, c=c, alpha=alpha, gamma=gamma, A=A, clip_norm=clip_norm
        )

    p = evaluator.p
    d = 2 * p
    ls_layers = min(ls_layers, p)
    mask = np.zeros(d, dtype=float)
    mask[p - ls_layers : p] = 1.0
    mask[p + (p - ls_layers) : 2 * p] = 1.0

    stage1 = spsa_optimize(
        evaluator,
        theta0,
        ls_steps,
        seed,
        a=a,
        c=c,
        alpha=alpha,
        gamma=gamma,
        A=A,
        clip_norm=clip_norm,
        update_mask=mask,
    )
    stage2 = spsa_optimize(
        evaluator,
        stage1.best_theta,
        iters - ls_steps,
        seed + 999,
        a=a,
        c=c,
        alpha=alpha,
        gamma=gamma,
        A=A,
        clip_norm=clip_norm,
        update_mask=None,
    )

    hist = stage1.history + stage2.history
    return OptTrace(best=stage2.best, best_theta=stage2.best_theta, history=hist)


def multistart_spsa(
    evaluator: QAOAEvaluator,
    restarts: int,
    iters: int,
    seed: int,
    a: float,
    c: float,
    alpha: float,
    gamma: float,
    A: float,
    clip_norm: Optional[float] = None,
) -> OptTrace:
    rng = np.random.default_rng(seed)
    best = -float("inf")
    best_theta = None
    best_hist = None

    for r in range(restarts):
        theta0 = random_theta(evaluator.p, rng)
        tr = spsa_optimize(
            evaluator,
            theta0,
            iters,
            seed + 10_000 * r,
            a=a,
            c=c,
            alpha=alpha,
            gamma=gamma,
            A=A,
            clip_norm=clip_norm,
        )
        if tr.best > best:
            best = tr.best
            best_theta = tr.best_theta
            best_hist = tr.history

    assert best_theta is not None
    return OptTrace(best=float(best), best_theta=best_theta, history=best_hist)


def multistart_spsa_pool(
    evaluator: QAOAEvaluator,
    restarts: int,
    iters: int,
    seed: int,
    a: float,
    c: float,
    alpha: float,
    gamma: float,
    A: float,
    clip_norm: Optional[float] = None,
) -> List[OptTrace]:
    """Run multiple SPSA restarts and return the per-restart best traces.

    This enables *transfer-aware theta selection*: the best donor optimum is
    not necessarily the most transferable basin on the target.
    """
    rng = np.random.default_rng(seed)
    traces: List[OptTrace] = []
    for r in range(restarts):
        theta0 = random_theta(evaluator.p, rng)
        tr = spsa_optimize(
            evaluator,
            theta0,
            iters,
            seed + 10_000 * r,
            a=a,
            c=c,
            alpha=alpha,
            gamma=gamma,
            A=A,
            clip_norm=clip_norm,
        )
        traces.append(tr)
    return traces


# ---------------------------
# Surrogate candidate families
# ---------------------------

@dataclass
class SurrogateCandidate:
    name: str
    W: np.ndarray


def scale_to_match(target_W: np.ndarray, cand_W: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    t = float(np.mean(np.abs(target_W[np.triu_indices_from(target_W, 1)])))
    c = float(np.mean(np.abs(cand_W[np.triu_indices_from(cand_W, 1)])))
    if c < eps:
        return cand_W.copy(), 1.0
    s = t / c
    return cand_W * s, s


def make_additive_powlaw(n: int, seed: int, alpha: float = 1.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(1, n + 1, dtype=float)
    P = idx ** (-alpha)
    rng.shuffle(P)
    W = P.reshape(-1, 1) + P.reshape(1, -1)
    np.fill_diagonal(W, 0.0)
    return np.maximum(W, 0.0)


def make_additive_strength(target_W: np.ndarray, seed: int, noise: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    strength = np.sum(target_W, axis=1)
    strength = strength - strength.min()
    if strength.max() > 0:
        strength = strength / strength.max()
    strength = strength + noise * rng.standard_normal(len(strength))
    strength = np.maximum(strength, 0.0)
    W = strength.reshape(-1, 1) + strength.reshape(1, -1)
    np.fill_diagonal(W, 0.0)
    return W


def make_block_dense(n: int, seed: int, B: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, B, size=n)
    M = rng.random((B, B))
    M = 0.5 * (M + M.T)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            w = M[groups[i], groups[j]]
            W[i, j] = w
            W[j, i] = w
    np.fill_diagonal(W, 0.0)
    return W


def make_circulant_dense(n: int, seed: int, k_freq: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    c = np.zeros(n)
    for k in range(1, k_freq + 1):
        amp = rng.random() * 0.5
        phase = rng.random() * 2 * math.pi
        c += amp * np.cos(2 * math.pi * k * t / n + phase)
    c = c - c.min()
    if c.max() > 0:
        c = c / c.max()
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            w = c[d]
            W[i, j] = w
            W[j, i] = w
    np.fill_diagonal(W, 0.0)
    return np.maximum(W, 0.0)


def make_lowrank_dense(n: int, seed: int, rank: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, rank))
    W = U @ U.T
    np.fill_diagonal(W, 0.0)
    W = W - W.min()
    if W.max() > 0:
        W = W / W.max()
    return np.maximum(W, 0.0)


def generate_candidates(target: WeightedGraph, n_cands: int, seed: int, families: List[str]) -> List[SurrogateCandidate]:
    rng = np.random.default_rng(seed)
    cands: List[SurrogateCandidate] = []

    for i in range(n_cands):
        fam = families[i % len(families)]
        s = int(rng.integers(0, 1_000_000_000))
        if fam == "powlaw":
            Wc = make_additive_powlaw(target.n, s)
        elif fam == "strength":
            Wc = make_additive_strength(target.W, s)
        elif fam == "block":
            B = 4 if target.n <= 16 else 6
            Wc = make_block_dense(target.n, s, B=B)
        elif fam == "circulant":
            Wc = make_circulant_dense(target.n, s)
        elif fam == "lowrank2":
            Wc = make_lowrank_dense(target.n, s, rank=2)
        else:
            raise ValueError(f"Unknown surrogate family: {fam}")

        Wc, _ = scale_to_match(target.W, Wc)
        Wc = np.maximum(Wc, 0.0)
        np.fill_diagonal(Wc, 0.0)
        cands.append(SurrogateCandidate(name=fam, W=Wc))

    return cands


# ---------------------------
# Fingerprint probes + directional derivatives
# ---------------------------

def make_theta_probes_global(
    p: int,
    k: int,
    seed: int,
    *,
    beta_max: float = 0.4 * math.pi,
    gamma_max: float = 0.7 * math.pi,
    noise_sigma: float = 0.05,
) -> List[np.ndarray]:
    """Global/random probes in a plausible QAOA parameter region."""
    rng = np.random.default_rng(seed)
    probes: List[np.ndarray] = []
    for _ in range(k):
        b0 = rng.uniform(0.0, beta_max, size=p) + rng.normal(0.0, noise_sigma, size=p)
        g0 = rng.uniform(0.0, gamma_max, size=p) + rng.normal(0.0, noise_sigma, size=p)
        probes.append(clip_theta(np.concatenate([b0, g0]), p))
    return probes


def make_theta_probes_mixed(
    p: int,
    k: int,
    seed: int,
    *,
    theta_center: Optional[np.ndarray] = None,
    local_frac: float = 0.0,
    local_sigma_beta: float = 0.08 * math.pi,
    local_sigma_gamma: float = 0.08 * math.pi,
    global_beta_max: float = 0.4 * math.pi,
    global_gamma_max: float = 0.7 * math.pi,
    global_noise_sigma: float = 0.05,
) -> List[np.ndarray]:
    """Mixed probes: some local around theta_center, some global/random."""
    local_frac = float(max(0.0, min(1.0, local_frac)))
    if theta_center is None or local_frac <= 0.0 or k <= 0:
        return make_theta_probes_global(
            p, k, seed, beta_max=global_beta_max, gamma_max=global_gamma_max, noise_sigma=global_noise_sigma
        )

    theta_center = np.asarray(theta_center, dtype=float).reshape(-1)
    if theta_center.shape[0] != 2 * p:
        raise ValueError(f"theta_center must have length 2p={2*p}, got {theta_center.shape[0]}")

    rng = np.random.default_rng(seed)
    k_local = int(round(k * local_frac))
    k_local = max(0, min(k, k_local))
    k_global = k - k_local

    probes: List[np.ndarray] = []
    for _ in range(k_local):
        b_c, g_c = split_theta(theta_center, p)
        b = b_c + rng.normal(0.0, local_sigma_beta, size=p)
        g = g_c + rng.normal(0.0, local_sigma_gamma, size=p)
        probes.append(clip_theta(np.concatenate([b, g]), p))

    if k_global > 0:
        probes.extend(
            make_theta_probes_global(
                p,
                k_global,
                seed + 101_001,
                beta_max=global_beta_max,
                gamma_max=global_gamma_max,
                noise_sigma=global_noise_sigma,
            )
        )
    rng.shuffle(probes)
    return probes


def make_theta_probes(p: int, k: int, seed: int) -> List[np.ndarray]:
    """Backward-compatible alias for global probes."""
    return make_theta_probes_global(p, k, seed)

def eval_values(evaluator: QAOAEvaluator, thetas: List[np.ndarray]) -> np.ndarray:
    return np.array([evaluator.expected_cut(th) for th in thetas], dtype=float)


def make_directions(d: int, k_dirs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.choice([-1.0, 1.0], size=(k_dirs, d)).astype(float)
    # normalize to unit length
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    return dirs


def eval_dir_derivs(evaluator: QAOAEvaluator, thetas: List[np.ndarray], dirs: np.ndarray, eps: float) -> np.ndarray:
    """Directional derivatives: for each theta and each direction v, compute
    (f(theta+eps v) - f(theta-eps v)) / (2 eps).

    Returns array shape (K, k_dirs).
    """
    out = np.zeros((len(thetas), dirs.shape[0]), dtype=float)
    p = evaluator.p
    for i, th in enumerate(thetas):
        th = clip_theta(th, p)
        for j, v in enumerate(dirs):
            f_plus = evaluator.expected_cut(clip_theta(th + eps * v, p))
            f_minus = evaluator.expected_cut(clip_theta(th - eps * v, p))
            out[i, j] = (f_plus - f_minus) / (2.0 * eps)
    return out


@dataclass
class Fingerprint:
    rho_val: float
    grad_dir_cos: float
    hit: float
    score: float


def fingerprint_score(
    target_vals: np.ndarray,
    cand_vals: np.ndarray,
    target_dd: np.ndarray,
    cand_dd: np.ndarray,
    topk: int,
    w_rho: float,
    w_grad: float,
    w_hit: float,
) -> Fingerprint:
    rho = spearman_r(target_vals, cand_vals)

    # flatten directional derivatives (K * k_dirs)
    tvec = target_dd.reshape(-1)
    cvec = cand_dd.reshape(-1)
    grad_cos = cosine_sim(tvec, cvec)

    t_best = int(np.argmax(target_vals))
    cand_rank = np.argsort(-cand_vals)
    hit = 1.0 if t_best in set(cand_rank[: max(1, topk)]) else 0.0

    score = 0.0
    score += w_rho * (0.0 if math.isnan(rho) else rho)
    score += w_grad * (0.0 if math.isnan(grad_cos) else grad_cos)
    score += w_hit * hit

    return Fingerprint(rho_val=float(rho), grad_dir_cos=float(grad_cos), hit=float(hit), score=float(score))


# ---------------------------
# Instance runner
# ---------------------------

@dataclass
class InstanceResult:
    family: str
    seed: int
    n: int
    p_target: int

    opt: float
    base: float
    tr: float
    warm: float
    randFT: float

    gain_warm: float
    gain_randFT: float

    # Reporting for scientific honesty
    # gain_cond: PASS-only gain (NaN if SKIP)
    # gain_uncond: unconditional gain with SKIP treated as 0.0
    gain_cond: float
    gain_uncond: float

    gate_pass: int

    gate: str
    gate_reason: str

    fp_rho: float
    fp_grad: float
    fp_hit: float
    fp_score: float

    donor: str
    scale: float

    # Best scaling chosen for transfer init (eval-only on target)
    beta_mult_best: float
    gamma_mult_best: float

    evals_baseline: int
    evals_randFT: int
    evals_fp_target: int
    evals_fp_cands: int
    evals_source: int
    evals_warm: int

    # Breakdown of evals for paper-grade accounting
    evals_target_base: int
    evals_target_randFT: int
    evals_target_overhead: int
    evals_target_warm: int
    evals_donor_opt: int

    evals_total: int

    time_total_s: float

    # Optional traces (best-so-far per iter), useful for plots
    trace_base: Optional[List[float]]
    trace_randFT: Optional[List[float]]
    trace_warm: Optional[List[float]]

    # Wall-time breakdown (seconds) for transparency
    wall_baseline_s: float
    wall_randFT_s: float
    wall_fp_target_s: float
    wall_fp_cands_s: float
    wall_source_s: float
    wall_warm_s: float


def brute_force_opt(graph: WeightedGraph) -> float:
    return float(np.max(graph.cut_value))


def run_instance(args, graph: WeightedGraph) -> InstanceResult:
    n = graph.n
    pT = args.p_target
    pM = args.p_match

    t_start = now()

    # Evaluator backend (numpy or torch). Torch can use GPU if CUDA is available.
    backend = getattr(args, "backend", "numpy")
    device = getattr(args, "device", "auto")
    torch_dtype = getattr(args, "torch_dtype", "complex64")
    torch_real_dtype = getattr(args, "torch_real_dtype", "float32")

    def E(g: WeightedGraph, p: int):
        return make_evaluator(g, p, backend=backend, device=device, dtype=torch_dtype, real_dtype=torch_real_dtype)

    # Opt
    opt = brute_force_opt(graph)

    # Baseline
    eval_target = E(graph, pT)
    eval_target.reset_stats()
    base_trace = multistart_spsa(
        eval_target,
        restarts=args.restarts_baseline,
        iters=args.iters,
        seed=graph.seed + 10_000,
        a=args.spsa_a,
        c=args.spsa_c,
        alpha=args.spsa_alpha,
        gamma=args.spsa_gamma,
        A=args.spsa_A,
        clip_norm=args.grad_clip,
    )
    base = base_trace.best
    base_theta = base_trace.best_theta

    # Fingerprint center: project target baseline (p_target) down to p_match.
    theta_base_fp: Optional[np.ndarray] = None
    if args.fp_local_frac > 0.0:
        theta_base_fp = lift_theta(base_theta, pT, pM, mode=args.lift_mode)

    evals_baseline = eval_target.stats.n_evals
    evals_target_base = int(evals_baseline)
    wall_baseline_s = eval_target.stats.wall_s
    trace_base = base_trace.history if args.save_traces else None
    wall_randFT_s = 0.0
    trace_randFT = None

    # RandFT: same budget as warm fine-tune, but from random init(s)
    eval_target.reset_stats()
    rng = np.random.default_rng(graph.seed + 20_000)
    best_randFT = -float("inf")
    best_rand_trace = None
    for rr in range(max(1, args.probe_rand_restarts)):
        theta0_rand = random_theta(pT, rng)
        trr = scheduled_spsa(
            eval_target,
            theta0_rand,
            iters=args.finetune_iters,
            seed=graph.seed + 21_000 + 999 * rr,
            a=args.spsa_a_ft,
            c=args.spsa_c_ft,
            alpha=args.spsa_alpha,
            gamma=args.spsa_gamma,
            A=args.spsa_A,
            ls_steps=args.layer_selective_steps,
            ls_layers=args.layer_selective_layers,
            clip_norm=args.grad_clip,
        )
        if trr.best > best_randFT:
            best_randFT = trr.best
            best_rand_trace = trr

    randFT = float(best_randFT)
    evals_rand = eval_target.stats.n_evals
    evals_target_randFT = int(evals_rand)
    wall_randFT_s = eval_target.stats.wall_s
    trace_randFT = best_rand_trace.history if (args.save_traces and best_rand_trace is not None) else None

    # Fingerprint probes (mixed local+global around baseline at p_match)
    p_fp = pM
    thetas_pre = make_theta_probes_mixed(
        p_fp,
        args.fp_pre_points,
        graph.seed + 30_000,
        theta_center=theta_base_fp,
        local_frac=args.fp_local_frac,
        local_sigma_beta=args.fp_local_sigma_beta,
        local_sigma_gamma=args.fp_local_sigma_gamma,
        global_beta_max=args.fp_global_beta_max,
        global_gamma_max=args.fp_global_gamma_max,
        global_noise_sigma=args.fp_noise_sigma,
    )
    thetas_full = make_theta_probes_mixed(
        p_fp,
        args.fp_points,
        graph.seed + 31_000,
        theta_center=theta_base_fp,
        local_frac=args.fp_local_frac,
        local_sigma_beta=args.fp_local_sigma_beta,
        local_sigma_gamma=args.fp_local_sigma_gamma,
        global_beta_max=args.fp_global_beta_max,
        global_gamma_max=args.fp_global_gamma_max,
        global_noise_sigma=args.fp_noise_sigma,
    )

    dirs = make_directions(d=2 * p_fp, k_dirs=args.fp_dirs, seed=graph.seed + 32_000)

    # Target fp values + dir-derivs
    eval_fp_target = E(graph, p_fp)
    eval_fp_target.reset_stats()
    target_vals_pre = eval_values(eval_fp_target, thetas_pre)
    target_vals_full = eval_values(eval_fp_target, thetas_full)
    target_dd_full = eval_dir_derivs(eval_fp_target, thetas_full, dirs, eps=args.fp_eps)
    evals_fp_target = eval_fp_target.stats.n_evals
    wall_fp_target_s = eval_fp_target.stats.wall_s

    # Candidates
    cand_fams = [s.strip() for s in args.surrogate_families.split(",") if s.strip()]
    cands = generate_candidates(graph, args.fp_cands, graph.seed + 40_000, cand_fams)

    fp_evals_cands = 0
    wall_fp_cands_s = 0.0

    # Pre-screen by rho on pre probes
    pre_scores = []
    for ci, cand in enumerate(cands):
        g_cand = WeightedGraph(n=n, W=cand.W, family=f"cand_{cand.name}", seed=graph.seed)
        g_cand.build_cache()
        eval_c = E(g_cand, p_fp)
        eval_c.reset_stats()
        cand_vals_pre = eval_values(eval_c, thetas_pre)
        rho_pre = spearman_r(target_vals_pre, cand_vals_pre)
        pre_scores.append((rho_pre if not math.isnan(rho_pre) else -1e9, ci))
        fp_evals_cands += eval_c.stats.n_evals
        wall_fp_cands_s += eval_c.stats.wall_s

    pre_scores.sort(reverse=True)
    top_idx = [ci for _, ci in pre_scores[: max(1, args.fp_preselect)]]

    # Full fingerprint (score many candidates)
    fp_records = []  # list of dicts: {score, fp, cand, scale}

    for ci in top_idx:
        cand = cands[ci]
        Wc, sc = scale_to_match(graph.W, cand.W)
        g_cand = WeightedGraph(n=n, W=Wc, family=f"cand_{cand.name}", seed=graph.seed)
        g_cand.build_cache()
        eval_c = E(g_cand, p_fp)
        eval_c.reset_stats()
        cand_vals_full = eval_values(eval_c, thetas_full)
        cand_dd_full = eval_dir_derivs(eval_c, thetas_full, dirs, eps=args.fp_eps)

        fp = fingerprint_score(
            target_vals_full,
            cand_vals_full,
            target_dd_full,
            cand_dd_full,
            topk=args.fp_topk,
            w_rho=args.fp_w_rho,
            w_grad=args.fp_w_grad,
            w_hit=args.fp_w_hit,
        )

        fp_evals_cands += eval_c.stats.n_evals
        wall_fp_cands_s += eval_c.stats.wall_s

        fp_records.append({
            "ci": int(ci),
            "score": float(fp.score),
            "fp": fp,
            "cand": SurrogateCandidate(name=cand.name, W=Wc),
            "scale": float(sc),
        })

    assert len(fp_records) > 0
    fp_records.sort(key=lambda r: r["score"], reverse=True)

    # Best-by-fingerprint (for debugging / reporting even if we later select a different donor)
    best_fp = fp_records[0]["fp"]
    best_cand = fp_records[0]["cand"]
    best_scale = fp_records[0]["scale"]

    # Gate 1: fingerprint (eligibility)
    pass_records = [
        r for r in fp_records
        if (r["fp"].rho_val >= args.gate_rho)
        and (r["fp"].grad_dir_cos >= args.gate_grad)
        and (r["fp"].hit >= args.gate_hit)
    ]

    # NEW: try top-K eligible donors using a cheap *probe* (donor SPSA + transfer eval)
    # This improves recall without sacrificing scientific honesty because Gate-2 still prevents negative transfer.
    # Evals accounting for paper-grade honesty
    evals_donor_probe = 0
    wall_donor_probe_s = 0.0
    evals_target_overhead_probe = 0
    wall_target_overhead_probe_s = 0.0

    gate_pass_fp = len(pass_records) > 0

    if gate_pass_fp and (args.try_topk_donors > 0) and (args.src_probe_iters > 0) and (args.src_probe_restarts > 0):
        k_try = int(args.try_topk_donors)
        # Diversify: ensure at least one donor per surrogate family in the probe set.
        # This reduces the chance that the probe selection collapses to a single family (e.g., always circulant).
        best_per_fam = {}
        for r in pass_records:
            fam = r["cand"].name
            if (fam not in best_per_fam) or (r["score"] > best_per_fam[fam]["score"]):
                best_per_fam[fam] = r
        try_records = list(best_per_fam.values())
        try_records.sort(key=lambda r: r["score"], reverse=True)
        if len(try_records) > k_try:
            try_records = try_records[:k_try]
        if len(try_records) < k_try:
            used_ci = set(r.get("ci", -1) for r in try_records)
            for r in pass_records:
                if r.get("ci", -2) in used_ci:
                    continue
                try_records.append(r)
                used_ci.add(r.get("ci", -2))
                if len(try_records) >= k_try:
                    break


        best_probe_tr = -float("inf")
        best_probe_rec = None

        # One target evaluator reused for probe transfer scoring
        eval_tr_probe = E(graph, pT)
        eval_tr_probe.reset_stats()

        # Parse scaling grid once (if enabled)
        beta_mults = [float(x) for x in args.beta_mults.split(",") if x.strip()]
        gamma_mults = [float(x) for x in args.gamma_mults.split(",") if x.strip()]
        warm_gamma_mults = [float(x) for x in args.warm_gamma_mults.split(",") if x.strip()]

        for ri, rec in enumerate(try_records):
            cand_rec = rec["cand"]

            # Probe donor optimization (cheap) + pool of candidate thetas.
            g_src_p = WeightedGraph(n=n, W=cand_rec.W, family=f"donorProbe_{cand_rec.name}", seed=graph.seed)
            g_src_p.build_cache()
            eval_src_p = E(g_src_p, pM)
            eval_src_p.reset_stats()
            src_p_pool = multistart_spsa_pool(
                eval_src_p,
                restarts=max(1, args.src_probe_restarts),
                iters=max(5, args.src_probe_iters),
                seed=graph.seed + 50_000 + 997 * ri,
                a=args.spsa_a_src,
                c=args.spsa_c_src,
                alpha=args.spsa_alpha,
                gamma=args.spsa_gamma,
                A=args.spsa_A,
                clip_norm=args.grad_clip,
            )
            src_p_pool.sort(key=lambda tr: tr.best, reverse=True)
            k_pool = max(1, min(int(args.src_probe_pool_k), len(src_p_pool)))
            src_p_pool = src_p_pool[:k_pool]

            evals_donor_probe += eval_src_p.stats.n_evals
            wall_donor_probe_s += eval_src_p.stats.wall_s

            # Transfer mapping: gamma scaling (same for the whole pool)
            num = float(np.mean(np.abs(graph.W[np.triu_indices(n, 1)])))
            den = float(np.mean(np.abs(cand_rec.W[np.triu_indices(n, 1)])) + 1e-12)
            gamma_scale = float(np.clip(num / den, 1.0 / args.gamma_clip, args.gamma_clip))

            # Probe transfer on target (eval-only), selecting best transferable theta from the pool
            best_init_val_p = -float("inf")
            for tr_src in src_p_pool:
                _, val_p, _, _ = transfer_and_select_init_eval_only(
                    eval_tr_probe,
                    tr_src.best_theta,
                    p_src=pM,
                    p_target=pT,
                    gamma_scale=gamma_scale,
                    lift_mode=args.lift_mode,
                    use_2d_scaling_grid=args.use_2d_scaling_grid,
                    beta_mults=beta_mults,
                    gamma_mults=gamma_mults,
                    warm_gamma_mults=warm_gamma_mults,
                )
                if val_p > best_init_val_p:
                    best_init_val_p = float(val_p)

            if best_init_val_p > best_probe_tr:
                best_probe_tr = float(best_init_val_p)
                best_probe_rec = rec

            # Early accept if already clearly above baseline
            if best_probe_tr >= (base + args.tr_accept_margin):
                break

        # Account probe target evals separately as *target overhead* (not donor work)
        evals_target_overhead_probe = int(eval_tr_probe.stats.n_evals)
        wall_target_overhead_probe_s = float(eval_tr_probe.stats.wall_s)

        # Select donor by best *probe* transfer score
        if best_probe_rec is not None:
            best_fp = best_probe_rec["fp"]
            best_cand = best_probe_rec["cand"]
            best_scale = best_probe_rec["scale"]

    donor_name = best_cand.name
    tr_val = float("nan")
    warm_val = float("nan")
    evals_source = 0  # donor-side evaluations (probe + full donor opt)
    evals_warm = 0    # legacy: target overhead + target warm (kept for backward compatibility)

    # Paper-grade eval breakdown (target overhead excluded from donor work)
    evals_target_overhead_transfer = 0
    evals_target_warm = 0

    # Best scaling chosen for transfer init
    beta_mult_best = 1.0
    gamma_mult_best = 1.0

    wall_source_s = 0.0
    wall_warm_s = 0.0
    trace_warm = None

    gate = "SKIP"
    gate_reason = "fp_gate_fail"

    if gate_pass_fp:
        # Source optimization (donor). We keep a pool of top donor optima and
        # select the one that transfers best (eval-only) to the target.
        g_src = WeightedGraph(n=n, W=best_cand.W, family=f"donor_{donor_name}", seed=graph.seed)
        g_src.build_cache()
        eval_src = E(g_src, pM)
        eval_src.reset_stats()

        src_pool = multistart_spsa_pool(
            eval_src,
            restarts=args.restarts_source,
            iters=args.src_iters,
            seed=graph.seed + 50_000,
            a=args.spsa_a_src,
            c=args.spsa_c_src,
            alpha=args.spsa_alpha,
            gamma=args.spsa_gamma,
            A=args.spsa_A,
            clip_norm=args.grad_clip,
        )
        src_pool.sort(key=lambda tr: tr.best, reverse=True)
        k_pool = max(1, min(int(args.transfer_pool_k), len(src_pool)))
        src_pool = src_pool[:k_pool]

        evals_source = int(evals_donor_probe + eval_src.stats.n_evals)
        wall_source_s = float(wall_donor_probe_s + eval_src.stats.wall_s)

        # Transfer scaling (shared)
        num = float(np.mean(np.abs(graph.W[np.triu_indices(n, 1)])))
        den = float(np.mean(np.abs(best_cand.W[np.triu_indices(n, 1)])) + 1e-12)
        gamma_scale = float(np.clip(num / den, 1.0 / args.gamma_clip, args.gamma_clip))

        beta_mults = [float(x) for x in args.beta_mults.split(",") if x.strip()]
        gamma_mults = [float(x) for x in args.gamma_mults.split(",") if x.strip()]
        warm_gamma_mults = [float(x) for x in args.warm_gamma_mults.split(",") if x.strip()]

        # Evaluate transfer on target and choose best init from the pool (eval-only)
        eval_tr = E(graph, pT)
        eval_tr.reset_stats()

        best_init = None
        best_init_val = -float("inf")
        best_bm = 1.0
        best_gm = 1.0

        for tr_src in src_pool:
            init_th, init_val, bm, gm = transfer_and_select_init_eval_only(
                eval_tr,
                tr_src.best_theta,
                p_src=pM,
                p_target=pT,
                gamma_scale=gamma_scale,
                lift_mode=args.lift_mode,
                use_2d_scaling_grid=args.use_2d_scaling_grid,
                beta_mults=beta_mults,
                gamma_mults=gamma_mults,
                warm_gamma_mults=warm_gamma_mults,
            )
            if init_val > best_init_val:
                best_init_val = float(init_val)
                best_init = init_th
                best_bm = float(bm)
                best_gm = float(gm)

            # Optional early accept to reduce overhead when transfer is clearly good
            if best_init_val >= (base + args.tr_accept_margin):
                break

        assert best_init is not None
        beta_mult_best = float(best_bm)
        gamma_mult_best = float(best_gm)

        # Use the best init value as the reported transfer score
        tr_val = float(best_init_val)
        evals_target_overhead_transfer = int(eval_tr.stats.n_evals)

        mw_evals = 0
        mw_wall_s = 0.0

        # Gate 2: transfer sanity (uses best init after gamma-mult search)
        if tr_val < base - args.transfer_margin:
            # Optional: micro-warm rescue when we're *very* close to base
            if args.micro_warm and (base - tr_val) <= args.micro_warm_margin:
                eval_mw = E(graph, pT)
                eval_mw.reset_stats()
                mw_trace = scheduled_spsa(
                    eval_mw,
                    best_init,
                    iters=args.micro_warm_iters,
                    seed=graph.seed + 59_000,
                    a=args.spsa_a_ft,
                    c=args.spsa_c_ft,
                    alpha=args.spsa_alpha,
                    gamma=args.spsa_gamma,
                    A=args.spsa_A,
                    ls_steps=min(args.layer_selective_steps, max(0, args.micro_warm_iters // 2)),
                    ls_layers=args.layer_selective_layers,
                    clip_norm=args.grad_clip,
                )
                tr_val = float(mw_trace.best)
                best_init = mw_trace.best_theta

                mw_evals = eval_mw.stats.n_evals
                mw_wall_s = eval_mw.stats.wall_s
                # Proceed to full warm from (possibly) micro-warmed init.
                eval_w = E(graph, pT)
                eval_w.reset_stats()
                warm_trace = scheduled_spsa(
                    eval_w,
                    best_init,
                    iters=args.finetune_iters,
                    seed=graph.seed + 60_000,
                    a=args.spsa_a_ft,
                    c=args.spsa_c_ft,
                    alpha=args.spsa_alpha,
                    gamma=args.spsa_gamma,
                    A=args.spsa_A,
                    ls_steps=args.layer_selective_steps,
                    ls_layers=args.layer_selective_layers,
                    clip_norm=args.grad_clip,
                )
                warm_val = float(max(tr_val, warm_trace.best))  # enforce monotonic improvement from init
                evals_warm = eval_tr.stats.n_evals + eval_mw.stats.n_evals + eval_w.stats.n_evals
                evals_target_warm = int(eval_mw.stats.n_evals + eval_w.stats.n_evals)
                wall_warm_s = eval_tr.stats.wall_s + eval_mw.stats.wall_s + eval_w.stats.wall_s
                gate = "PASS"
                gate_reason = "pass"

                if args.save_traces:
                    trace_warm = warm_trace.history
            else:
                gate = "SKIP"
                gate_reason = "transfer_worse_than_base"
                evals_warm = eval_tr.stats.n_evals
                evals_target_warm = 0
                wall_warm_s = eval_tr.stats.wall_s
                warm_val = float("nan")
        else:
            # Fine-tune (target) with scheduled SPSA
            eval_w = E(graph, pT)
            eval_w.reset_stats()
            warm_trace = scheduled_spsa(
                eval_w,
                best_init,
                iters=args.finetune_iters,
                seed=graph.seed + 60_000,
                a=args.spsa_a_ft,
                c=args.spsa_c_ft,
                alpha=args.spsa_alpha,
                gamma=args.spsa_gamma,
                A=args.spsa_A,
                ls_steps=args.layer_selective_steps,
                ls_layers=args.layer_selective_layers,
                clip_norm=args.grad_clip,
            )
            warm_val = float(max(tr_val, warm_trace.best))  # enforce monotonic improvement from init
            evals_warm = eval_tr.stats.n_evals + mw_evals + eval_w.stats.n_evals
            evals_target_warm = int(mw_evals + eval_w.stats.n_evals)
            wall_warm_s = eval_tr.stats.wall_s + mw_wall_s + eval_w.stats.wall_s
            gate = "PASS"
            gate_reason = "pass"

            if args.save_traces:
                trace_warm = warm_trace.history
    gate_pass = 1 if gate == "PASS" else 0
    gain_cond = float(warm_val - base) if gate_pass else float("nan")
    gain_uncond = float(warm_val - base) if gate_pass else 0.0
    gain_warm = gain_cond
    gain_rand = float(randFT - base)

    # Target overhead = donor-probe transfer scoring (on target) + transfer init scaling search (on target)
    evals_target_overhead = int(evals_target_overhead_probe + evals_target_overhead_transfer)

    # Donor optimization evals (probe + full)
    evals_donor_opt = int(evals_source)

    # Paper-grade total: include everything, but avoid double-counting target overhead inside warm.
    evals_total = int(
        evals_target_base
        + evals_target_randFT
        + evals_fp_target
        + fp_evals_cands
        + evals_donor_opt
        + evals_target_overhead
        + evals_target_warm
    )
    time_total = now() - t_start

    return InstanceResult(
        family=graph.family,
        seed=graph.seed,
        n=n,
        p_target=pT,
        opt=float(opt),
        base=float(base),
        tr=float(tr_val) if not math.isnan(tr_val) else float("nan"),
        warm=float(warm_val) if not math.isnan(warm_val) else float("nan"),
        randFT=float(randFT),
        gain_warm=gain_warm,
        gain_randFT=gain_rand,
        gain_cond=float(gain_cond),
        gain_uncond=float(gain_uncond),
        gate_pass=int(gate_pass),
        gate=gate,
        gate_reason=gate_reason,
        fp_rho=float(best_fp.rho_val),
        fp_grad=float(best_fp.grad_dir_cos),
        fp_hit=float(best_fp.hit),
        fp_score=float(best_fp.score),
        donor=donor_name,
        scale=float(best_scale),
        beta_mult_best=float(beta_mult_best),
        gamma_mult_best=float(gamma_mult_best),
        evals_baseline=int(evals_baseline),
        evals_randFT=int(evals_rand),
        evals_fp_target=int(evals_fp_target),
        evals_fp_cands=int(fp_evals_cands),
        evals_source=int(evals_source),
        evals_warm=int(evals_warm),
        evals_target_base=int(evals_target_base),
        evals_target_randFT=int(evals_target_randFT),
        evals_target_overhead=int(evals_target_overhead),
        evals_target_warm=int(evals_target_warm),
        evals_donor_opt=int(evals_donor_opt),
        evals_total=int(evals_total),
        time_total_s=float(time_total),
        trace_base=trace_base,
        trace_randFT=trace_randFT,
        trace_warm=trace_warm,
        wall_baseline_s=float(wall_baseline_s),
        wall_randFT_s=float(wall_randFT_s),
        wall_fp_target_s=float(wall_fp_target_s),
        wall_fp_cands_s=float(wall_fp_cands_s),
        wall_source_s=float(wall_source_s),
        wall_warm_s=float(wall_warm_s),
    )


# ---------------------------
# Saving / summary
# ---------------------------

def safe_mean(xs: List[float]) -> float:
    ys = [x for x in xs if x == x and not math.isinf(x)]
    return float(np.mean(ys)) if ys else float("nan")


def write_summary(results: List[InstanceResult], out_csv: str) -> None:
    # Group by (family, n, p_target) so sweeps produce structured summaries.
    groups = sorted(set((r.family, int(r.n), int(r.p_target)) for r in results))
    lines = [
        "family,n,p,fp_rho,fp_grad,fp_hit,fp_score,pass_rate,gain_warm_uncond,gain_warm_cond,gain_randFT,"
        "ratio_base,ratio_warm_cond,evals_total,evals_target_overhead,evals_target_warm,evals_donor_opt,time_total_s"
    ]
    for fam, n, p in groups:
        rs = [r for r in results if r.family == fam and int(r.n) == n and int(r.p_target) == p]
        fp_rho = safe_mean([r.fp_rho for r in rs])
        fp_grad = safe_mean([r.fp_grad for r in rs])
        fp_hit = safe_mean([r.fp_hit for r in rs])
        fp_score = safe_mean([r.fp_score for r in rs])
        pass_rate = safe_mean([float(r.gate_pass) for r in rs])
        gain_warm_uncond = safe_mean([r.gain_uncond for r in rs])
        gain_warm_cond = safe_mean([r.gain_cond for r in rs])
        gain_rand = safe_mean([r.gain_randFT for r in rs])
        ratio_base = safe_mean([r.base / r.opt for r in rs])
        ratio_warm = safe_mean([r.warm / r.opt for r in rs if r.gate == "PASS"]) 
        evals_total = safe_mean([float(r.evals_total) for r in rs])
        evals_target_overhead = safe_mean([float(r.evals_target_overhead) for r in rs])
        evals_target_warm = safe_mean([float(r.evals_target_warm) for r in rs])
        evals_donor_opt = safe_mean([float(r.evals_donor_opt) for r in rs])
        time_total = safe_mean([r.time_total_s for r in rs])
        lines.append(
            f"{fam},{n},{p},{fp_rho:.6f},{fp_grad:.6f},{fp_hit:.6f},{fp_score:.6f},{pass_rate:.3f},"
            f"{gain_warm_uncond:.6f},{gain_warm_cond:.6f},{gain_rand:.6f},{ratio_base:.6f},{ratio_warm:.6f},"
            f"{evals_total:.1f},{evals_target_overhead:.1f},{evals_target_warm:.1f},{evals_donor_opt:.1f},{time_total:.3f}"
        )

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _parse_int_csv(s: str) -> List[int]:
    """Parse comma-separated integers. Empty string => []."""
    s = (s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _apply_auto_budget(args, n: int, p: int) -> None:
    """Heuristic, paper-friendly budgets tuned for GPU sweeps.

    Goals:
      (1) keep compute manageable for n up to 20 and p up to 5,
      (2) keep methodology unchanged (same gates + same baselines),
      (3) improve recall by modestly increasing diversity where it matters.
    """
    n = int(n)
    p = int(p)

    # Baseline budget
    if n <= 14:
        args.restarts_baseline = max(args.restarts_baseline, 8)
        args.iters = max(args.iters, 140 if p <= 3 else 170)
        args.finetune_iters = max(args.finetune_iters, 160 if p <= 3 else 200)
        args.restarts_source = max(args.restarts_source, 6)
        args.src_iters = max(args.src_iters, 140 if p <= 3 else 170)
        args.probe_rand_restarts = max(args.probe_rand_restarts, 4)
        args.transfer_pool_k = max(args.transfer_pool_k, 6)
        args.fp_points = max(args.fp_points, 14)
        args.fp_cands = max(args.fp_cands, 70)
    elif n <= 18:
        args.restarts_baseline = max(args.restarts_baseline, 6)
        args.iters = max(args.iters, 110 if p <= 3 else 140)
        args.finetune_iters = max(args.finetune_iters, 140 if p <= 3 else 170)
        args.restarts_source = max(args.restarts_source, 5)
        args.src_iters = max(args.src_iters, 110 if p <= 3 else 140)
        args.probe_rand_restarts = max(args.probe_rand_restarts, 3)
        args.transfer_pool_k = max(args.transfer_pool_k, 5)
        args.fp_points = max(args.fp_points, 12)
        args.fp_cands = max(args.fp_cands, 55)
    else:  # n >= 20
        args.restarts_baseline = max(args.restarts_baseline, 5)
        args.iters = max(args.iters, 90 if p <= 3 else 120)
        args.finetune_iters = max(args.finetune_iters, 120 if p <= 3 else 150)
        args.restarts_source = max(args.restarts_source, 4)
        args.src_iters = max(args.src_iters, 90 if p <= 3 else 120)
        args.probe_rand_restarts = max(args.probe_rand_restarts, 3)
        args.transfer_pool_k = max(args.transfer_pool_k, 4)
        args.fp_points = max(args.fp_points, 10)
        args.fp_cands = max(args.fp_cands, 45)

    # Selection/recall knobs (still honest: Gate-2 stays in place)
    args.try_topk_donors = max(args.try_topk_donors, 2)
    args.use_2d_scaling_grid = True
    args.micro_warm = True



def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", default="maxcut", choices=["maxcut"])

    # Compatibility flags (ignored)
    ap.add_argument("--no_lightning", action="store_true", help="compat flag (ignored)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--p_target", type=int, default=3)
    ap.add_argument("--p_match", type=int, default=3)
    ap.add_argument("--lift_mode", type=str, default="repeat_last", choices=["repeat_last", "repeat", "pad_zero"],
                    help="how to lift p_match params to p_target")

    # Sweep helpers (comma-separated lists). If provided, the script runs all combinations.
    ap.add_argument("--n_values", type=str, default="",
                    help="Comma-separated n values to sweep, e.g. 12,14,18,20. Empty => use --n only.")
    ap.add_argument("--p_values", type=str, default="",
                    help="Comma-separated p_target values to sweep, e.g. 2,3,4,5. Empty => use --p_target only.")

    # Backend (numpy or torch). torch can run on GPU if CUDA is available.
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "numpy", "torch"],
                    help="QAOA evaluator backend. 'auto' => torch if CUDA available else numpy.")
    ap.add_argument("--device", type=str, default="auto",
                    help="Torch device: auto|cpu|cuda|cuda:0|... (ignored for numpy).")
    ap.add_argument("--torch_dtype", type=str, default="complex64", choices=["complex64", "complex128"],
                    help="Torch complex dtype for the statevector.")
    ap.add_argument("--torch_real_dtype", type=str, default="float32", choices=["float32", "float64"],
                    help="Torch real dtype for energies / probabilities.")

    # Optional auto-budget: in sweep mode we enable it by default unless you pass --no_auto_budget.
    ap.add_argument("--auto_budget", action="store_true", default=False,
                    help="Enable adaptive budgets as a function of (n,p) for sweeps.")
    ap.add_argument("--no_auto_budget", action="store_true", default=False,
                    help="Disable auto-budget even in sweep mode.")

    ap.add_argument("--families", type=str, default="ER_dense_p05_weighted,RR_3regular")
    ap.add_argument("--seeds_per_family", type=int, default=2)

    # Baseline SPSA
    ap.add_argument("--restarts_baseline", type=int, default=6)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--spsa_a", type=float, default=0.25)
    ap.add_argument("--spsa_c", type=float, default=0.12)

    # Source SPSA
    ap.add_argument("--restarts_source", type=int, default=4)
    ap.add_argument("--transfer_pool_k", type=int, default=5,
                    help="From the donor restarts, evaluate transfer on the best K donor optima (pool).")
    ap.add_argument("--src_iters", type=int, default=120)
    ap.add_argument("--spsa_a_src", type=float, default=0.25)
    ap.add_argument("--spsa_c_src", type=float, default=0.12)

    # Fine-tune SPSA
    ap.add_argument("--finetune_iters", type=int, default=140)
    ap.add_argument("--probe_rand_restarts", type=int, default=3, help="random restarts for randFT baseline")
    ap.add_argument("--spsa_a_ft", type=float, default=0.20)
    ap.add_argument("--spsa_c_ft", type=float, default=0.10)

    # Common SPSA schedule
    ap.add_argument("--spsa_alpha", type=float, default=0.602)
    ap.add_argument("--spsa_gamma", type=float, default=0.101)
    ap.add_argument("--spsa_A", type=float, default=10.0)

    ap.add_argument("--grad_clip", type=float, default=5.0)

    # Fingerprint
    ap.add_argument("--fp_points", type=int, default=12)
    ap.add_argument("--fp_pre_points", type=int, default=4)
    ap.add_argument("--fp_cands", type=int, default=40)
    ap.add_argument("--fp_preselect", type=int, default=12)
    ap.add_argument("--fp_topk", type=int, default=2)
    # Donor selection: try multiple top candidates (by fp score) and pick best transfer
    ap.add_argument("--try_topk_donors", type=int, default=0,
                    help="Try up to this many top fingerprint candidates (after fp) when selecting the donor.")
    ap.add_argument("--src_probe_iters", type=int, default=35,
                    help="Lightweight SPSA iterations for donor probing (cheap screening).")
    ap.add_argument("--src_probe_restarts", type=int, default=2,
                    help="Random restarts for donor probing (cheap screening).")
    ap.add_argument("--src_probe_pool_k", type=int, default=2,
                    help="From the donor-probe restarts, evaluate transfer on the best K donor optima (pool).")
    ap.add_argument("--tr_accept_margin", type=float, default=0.0,
                    help="Optional margin: accept early if probe transfer >= base + margin.")
    ap.add_argument("--fp_dirs", type=int, default=3)
    ap.add_argument("--fp_eps", type=float, default=2e-3)
    ap.add_argument("--fp_random_perms", type=int, default=0, help="compat flag (ignored)")
    ap.add_argument("--fp_w_rho", type=float, default=0.45)
    ap.add_argument("--fp_w_grad", type=float, default=0.45)
    ap.add_argument("--fp_w_hit", type=float, default=0.10)

    ap.add_argument("--fp_noise_sigma", type=float, default=0.05,
                    help="Gaussian noise sigma for global fingerprint probes (radians).")
    ap.add_argument("--fp_local_frac", type=float, default=0.5,
                    help="Fraction of fingerprint probes sampled locally around the target baseline (at p_match). 0 disables local probes.")
    ap.add_argument("--fp_local_sigma_beta", type=float, default=0.08 * math.pi,
                    help="Stddev (radians) for beta noise for local fingerprint probes.")
    ap.add_argument("--fp_local_sigma_gamma", type=float, default=0.08 * math.pi,
                    help="Stddev (radians) for gamma noise for local fingerprint probes.")
    ap.add_argument("--fp_global_beta_max", type=float, default=0.4 * math.pi,
                    help="Max beta range for global fingerprint probes (radians).")
    ap.add_argument("--fp_global_gamma_max", type=float, default=0.7 * math.pi,
                    help="Max gamma range for global fingerprint probes (radians).")

    # Gates
    ap.add_argument("--gate_rho", type=float, default=0.55)
    ap.add_argument("--gate_grad", type=float, default=0.25)
    ap.add_argument("--gate_hit", type=float, default=1.0)
    ap.add_argument("--transfer_margin", type=float, default=0.0)

    ap.add_argument("--micro_warm", action="store_true",
                    help="If tr is slightly worse than base, run a short SPSA test to try to cross base safely before skipping.")
    ap.add_argument("--micro_warm_margin", type=float, default=0.25,
                    help="Trigger micro-warm when (base - tr_best) <= margin.")
    ap.add_argument("--micro_warm_iters", type=int, default=30,
                    help="SPSA iterations for micro-warm test.")

    ap.add_argument("--surrogate_families", type=str, default="powlaw,strength,block,circulant,lowrank2")

    # Transfer-init eval-only scaling (paper-friendly): small grid over (beta_mult, gamma_mult)
    ap.add_argument("--use_2d_scaling_grid", action="store_true",
                    help="If set, replace 1D gamma-mult search with a small (beta,gamma) scaling grid (eval-only).")
    ap.add_argument("--beta_mults", type=str, default="0.9,1.0,1.1",
                    help="Comma-separated beta multipliers for 2D scaling grid (eval-only).")
    ap.add_argument("--gamma_mults", type=str, default="0.9,1.0,1.1",
                    help="Comma-separated gamma multipliers for 2D scaling grid (eval-only).")

    ap.add_argument("--warm_gamma_mults", type=str, default="1.0,0.95,1.05")

    ap.add_argument("--layer_selective_steps", type=int, default=40)
    ap.add_argument("--layer_selective_layers", type=int, default=1)

    ap.add_argument("--gamma_clip", type=float, default=2.3)
    ap.add_argument("--save_dir", type=str, default="./run_v7_3")
    ap.add_argument("--save_traces", action="store_true", help="store optimization traces in results_all.json")

    args = ap.parse_args()

    # Resolve backend='auto'
    if getattr(args, "backend", "auto") == "auto":
        if (torch is not None) and torch.cuda.is_available():
            args.backend = "torch"
        else:
            args.backend = "numpy"

    # Sweep settings
    n_vals = _parse_int_csv(args.n_values)
    p_vals = _parse_int_csv(args.p_values)
    sweep_mode = bool(n_vals) or bool(p_vals)
    if not n_vals:
        n_vals = [int(args.n)]
    if not p_vals:
        p_vals = [int(args.p_target)]

    # In sweep mode, default to auto-budget unless explicitly disabled.
    auto_budget = bool(args.auto_budget) or (sweep_mode and not bool(args.no_auto_budget))

    os.makedirs(args.save_dir, exist_ok=True)
    fams = [s.strip() for s in args.families.split(",") if s.strip()]

    all_results: List[InstanceResult] = []
    suite_t0 = now()

    for n in n_vals:
        for pT in p_vals:
            args_run = copy.copy(args)
            args_run.n = int(n)
            args_run.p_target = int(pT)
            # For fairness in sweeps: align p_match with p_target per run.
            args_run.p_match = int(pT)

            if auto_budget:
                _apply_auto_budget(args_run, n=int(n), p=int(pT))

            subdir = args_run.save_dir
            if sweep_mode:
                subdir = os.path.join(args_run.save_dir, f"n{int(n)}_p{int(pT)}")
                os.makedirs(subdir, exist_ok=True)

            results: List[InstanceResult] = []
            run_t0 = now()

            for fam in fams:
                for k in range(args_run.seeds_per_family):
                    seed = 42 + k if fam.startswith("Complete") else (1042 + k if fam.startswith("ER") else 2042 + k)
                    g = build_graph(fam, int(n), seed)

                    inst_t0 = now()
                    res = run_instance(args_run, g)
                    dt = now() - inst_t0
                    results.append(res)
                    all_results.append(res)

                    if res.gate == "PASS":
                        print(
                            f"[{res.family}|n={res.n}|seed={res.seed}|p={res.p_target}] opt={res.opt:.3f} base={res.base:.3f} "
                            f"tr={res.tr:.3f} warm={res.warm:.3f} randFT={res.randFT:.3f} "
                            f"gain(warm-base)={res.gain_warm:+.3f} fp=(rho={res.fp_rho:+.2f},grad={res.fp_grad:+.2f},hit={int(res.fp_hit)}) "
                            f"gate=PASS donor={res.donor} scale={res.scale:.3f} time={dt:.2f}s"
                        )
                    else:
                        tr_s = "nan" if (res.tr != res.tr) else f"{res.tr:.3f}"
                        print(
                            f"[{res.family}|n={res.n}|seed={res.seed}|p={res.p_target}] opt={res.opt:.3f} base={res.base:.3f} "
                            f"tr={tr_s} warm=nan randFT={res.randFT:.3f} "
                            f"gain(warm-base)=nan fp=(rho={res.fp_rho:+.2f},grad={res.fp_grad:+.2f},hit={int(res.fp_hit)}) "
                            f"gate=SKIP reason={res.gate_reason} donor={res.donor} scale={res.scale:.3f} time={dt:.2f}s"
                        )

            run_dt = now() - run_t0
            out_json = os.path.join(subdir, "results_all.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump([r.__dict__ for r in results], f, indent=2)
            out_csv = os.path.join(subdir, "summary.csv")
            write_summary(results, out_csv)

            print(f"\n✅ Run done for n={int(n)}, p={int(pT)}. Time(s) = {run_dt:.2f}")
            print(f"Saved: {out_json}")
            print(f"Saved: {out_csv}\n")

    suite_dt = now() - suite_t0

    # If sweep => write combined outputs in root save_dir for convenience.
    if sweep_mode and (len(n_vals) * len(p_vals) > 1):
        out_json = os.path.join(args.save_dir, "results_all_sweep.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([r.__dict__ for r in all_results], f, indent=2)
        out_csv = os.path.join(args.save_dir, "summary_sweep.csv")
        write_summary(all_results, out_csv)
        print(f"\n✅ Sweep done. Time(s) = {suite_dt:.2f}")
        print(f"Saved: {out_json}")
        print(f"Saved: {out_csv}")
    else:
        print(f"\n✅ Suite done. Time(s) = {suite_dt:.2f}")


if __name__ == "__main__":
    main()
