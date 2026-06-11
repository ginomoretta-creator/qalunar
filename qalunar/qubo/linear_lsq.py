"""Linear least-squares to QUBO transcription.

Port of Gino Moretta's MATLAB ``Rendezvous_GIno.m`` v2 pipeline, lifted
out of its rendezvous-specific context into a reusable primitive. Given
any overdetermined linear system ``A @ xi ~= B``, build a QUBO whose
ground state approximates the Tikhonov-regularized least-squares
solution under a per-variable adaptive binary encoding.

Pipeline
--------

1. **Tikhonov regularization.** Stack the regularizer onto the system
   so the least-squares problem becomes ``min ||A_aug @ xi - B_aug||^2``
   with

       A_aug = [A; sqrt(lambda) * I_N],   B_aug = [B; 0]

   and ``lambda = tikhonov_relative * sigma_max(A)^2``.

2. **SVD rank reduction.** Compute the thin SVD
   ``A_aug = U @ diag(s) @ V^T`` and keep the columns corresponding to
   singular values ``s_i > svd_threshold * s_0``. The unknowns are
   reparameterized in the truncated right basis, ``xi = V_r @ alpha``,
   so the reduced system is ``A_aug @ V_r @ alpha ~= B_aug``. The
   Tikhonov solution in reduced coordinates is
   ``alpha_tik = S_r^-1 @ U_r^T @ B_aug``.

3. **Per-variable adaptive encoding.** Each reduced coordinate
   ``alpha_i`` is quantized with ``p`` bits over an interval centered
   on ``alpha_tik_i``. The half-range is

       R_i = margin * (|alpha_tik_i| + ||res_aug|| / s_i)

   where ``s_i`` is the ``i``-th singular value of ``A_aug`` (so
   ``1 / s_i`` is the sensitivity of ``alpha_i`` to residual noise),
   and ``res_aug = B_aug - A_aug @ xi_tik``. The encoding is
   ``alpha = alpha_offset + D_dec @ q`` with ``D_dec`` block-diagonal
   in blocks ``step_i * [1, 2, 4, ..., 2^(p-1)]``.

4. **QUBO assembly.** Substituting the encoding into
   ``||A_aug @ V_r @ alpha - B_aug||^2`` yields a quadratic form
   in ``q`` with

       Phi     = A_red @ D_dec,   A_red = A_aug @ V_r
       B_tilde = B_aug - A_red @ alpha_offset
       Q       = Phi^T @ Phi
       linear  = -2 * Phi^T @ B_tilde
       const   = ||B_tilde||^2

   so ``E(q) = q^T Q q + linear^T q + const = ||A_aug @ xi(q) - B_aug||^2``
   *exactly*, which is the identity we use as a unit test.

References
----------
* Gino Moretta, ``Rendezvous_GIno.m`` (QUBO v2), Sapienza Rome, 2025.
* De Grossi, Carbone et al., *Pseudospectral QUBO transcription for
  low-thrust trajectory optimization*, Astrodynamics 9:195-215, 2025.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LinearLsqToQuboConfig:
    """Hyperparameters for :func:`build_linear_lsq_qubo`.

    Parameters
    ----------
    bits_per_variable : int, default 8
        Number of qubits per reduced coordinate ``alpha_i``. Total QUBO
        size is ``n_reduced * bits_per_variable``.
    tikhonov_relative : float, default 1e-6
        Regularization strength expressed as a fraction of
        ``sigma_max(A)^2``. Yields ``lambda = tikhonov_relative * sigma_max^2``
        so the numerical scale tracks the problem automatically.
    svd_threshold : float, default 1e-3
        Relative singular-value cutoff for rank truncation of
        ``A_aug``: keep ``s_i > svd_threshold * s_0``.
    encoding_margin : float, default 3.0
        Safety factor on the per-variable encoding range. A value of
        ``3.0`` means the half-range is three times the magnitude-plus-
        sensitivity estimate, i.e. roughly a "3 sigma" box around
        ``alpha_tik``.
    min_spread : float, default 1e-6
        Floor on the per-variable half-range to avoid degenerate
        zero-width intervals when both ``alpha_tik_i`` and the
        augmented residual are near machine precision.
    """

    bits_per_variable: int = 8
    tikhonov_relative: float = 1e-6
    svd_threshold: float = 1e-3
    encoding_margin: float = 3.0
    min_spread: float = 1e-6


@dataclass(frozen=True)
class LinearLsqToQuboResult:
    """Bundle returned by :func:`build_linear_lsq_qubo`.

    Carries the QUBO triple ``(Q, linear, const)`` along with everything
    needed to decode a bitstring back to the original unknowns ``xi``
    and a few diagnostics (singular spectrum, effective rank, Tikhonov
    reference solution).
    """

    # QUBO: E(q) = q^T Q q + linear^T q + const
    Q: NDArray[np.float64]
    linear: NDArray[np.float64]
    const: float

    # Decoding path: alpha = alpha_offset + step_sizes * int(q); xi = V_r @ alpha
    V_r: NDArray[np.float64]
    alpha_offset: NDArray[np.float64]
    step_sizes: NDArray[np.float64]
    bits_per_variable: int

    # Diagnostics / reference values
    alpha_tik: NDArray[np.float64]
    xi_tik: NDArray[np.float64]
    singular_values_augmented: NDArray[np.float64]
    effective_rank: int
    lambda_tikhonov: float
    config: LinearLsqToQuboConfig

    @property
    def n_variables(self) -> int:
        """Original problem dimension (columns of ``A``)."""
        return int(self.V_r.shape[0])

    @property
    def n_reduced(self) -> int:
        """Number of SVD-truncated unknowns ``alpha``."""
        return int(self.V_r.shape[1])

    @property
    def n_bits(self) -> int:
        """Total number of binary variables in the QUBO."""
        return int(self.n_reduced * self.bits_per_variable)

    def decode_alpha(self, q: NDArray[np.int_]) -> NDArray[np.float64]:
        """Decode a bitstring into the reduced coordinates ``alpha``."""
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self.n_bits,):
            raise ValueError(
                f"expected bitstring of length {self.n_bits}, got shape {q_arr.shape}"
            )
        p = self.bits_per_variable
        powers = (1 << np.arange(p)).astype(np.float64)  # [1, 2, 4, ..., 2^(p-1)]
        q_blocks = q_arr.reshape(self.n_reduced, p)
        integer_vals = q_blocks @ powers
        return self.alpha_offset + self.step_sizes * integer_vals

    def decode_xi(self, q: NDArray[np.int_]) -> NDArray[np.float64]:
        """Decode a bitstring into the original unknowns ``xi = V_r @ alpha(q)``."""
        return self.V_r @ self.decode_alpha(q)

    def energy(self, q: NDArray[np.int_]) -> float:
        """Evaluate the QUBO energy ``q^T Q q + linear^T q + const``."""
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self.n_bits,):
            raise ValueError(
                f"expected bitstring of length {self.n_bits}, got shape {q_arr.shape}"
            )
        return float(q_arr @ self.Q @ q_arr + self.linear @ q_arr + self.const)

    def warm_start_bits(self) -> NDArray[np.int64]:
        """Quantize ``alpha_tik`` to ``p`` bits as a warm-start bitstring.

        By construction the Tikhonov solution sits at the centre of the
        encoding box, so the bits correspond to the integer
        ``round((alpha_tik - alpha_offset) / step)`` clipped to
        ``[0, 2^p - 1]``. The decoded ``alpha`` is within half a step
        of ``alpha_tik`` per coordinate.
        """
        p = self.bits_per_variable
        max_int = (1 << p) - 1
        alpha_int = np.round(
            (self.alpha_tik - self.alpha_offset) / self.step_sizes
        )
        alpha_int = np.clip(alpha_int, 0, max_int).astype(np.int64)
        bits = np.zeros(self.n_bits, dtype=np.int64)
        for i in range(self.n_reduced):
            val = int(alpha_int[i])
            for k in range(p):
                bits[i * p + k] = (val >> k) & 1
        return bits


def build_linear_lsq_qubo(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    config: LinearLsqToQuboConfig | None = None,
) -> LinearLsqToQuboResult:
    """Build a QUBO encoding the Tikhonov-regularized LSQ solution of ``A @ xi = B``.

    See module docstring for the full pipeline.

    Parameters
    ----------
    A : (M, N) ndarray
        System matrix. Must be two-dimensional.
    B : (M,) ndarray
        Right-hand side. Must match ``A.shape[0]``.
    config : LinearLsqToQuboConfig, optional
        Hyperparameters controlling Tikhonov strength, SVD rank
        truncation, and per-variable encoding spread. Defaults are
        copied from ``Rendezvous_GIno.m``.

    Returns
    -------
    LinearLsqToQuboResult
        QUBO triple plus decode path and diagnostics.
    """
    cfg = config if config is not None else LinearLsqToQuboConfig()

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    if B.ndim != 1 or B.shape[0] != A.shape[0]:
        raise ValueError(
            f"B must be 1D with length A.shape[0]={A.shape[0]}, got shape {B.shape}"
        )
    if cfg.bits_per_variable < 1:
        raise ValueError(
            f"bits_per_variable must be >= 1, got {cfg.bits_per_variable}"
        )

    n_rows, n_vars = A.shape

    # ------------------------------------------------------------------
    # Step 1 - Tikhonov regularization
    # ------------------------------------------------------------------
    sigma_max_A = float(np.linalg.svd(A, compute_uv=False)[0])
    if not np.isfinite(sigma_max_A) or sigma_max_A <= 0.0:
        raise ValueError("A has zero or non-finite largest singular value")
    lambda_tik = cfg.tikhonov_relative * sigma_max_A ** 2
    sqrt_lambda = np.sqrt(lambda_tik)

    A_aug = np.vstack([A, sqrt_lambda * np.eye(n_vars)])
    B_aug = np.concatenate([B, np.zeros(n_vars)])

    # ------------------------------------------------------------------
    # Step 2 - SVD of the augmented system and rank truncation
    # ------------------------------------------------------------------
    U_aug, s_aug, Vt_aug = np.linalg.svd(A_aug, full_matrices=False)

    sigma_max_aug = float(s_aug[0])
    if sigma_max_aug <= 0.0:
        raise ValueError("augmented system has zero largest singular value")
    rank = int(np.sum(s_aug > cfg.svd_threshold * sigma_max_aug))
    if rank == 0:
        raise ValueError("svd_threshold truncated all singular values")

    U_r = U_aug[:, :rank]             # (M + N, r)
    s_r = s_aug[:rank]                # (r,)
    V_r = Vt_aug[:rank, :].T          # (N, r)

    # Tikhonov solution in reduced coordinates. S_r is diagonal, so
    # the solve is an element-wise division.
    alpha_tik = (U_r.T @ B_aug) / s_r
    xi_tik = V_r @ alpha_tik

    # Augmented residual drives the per-variable sensitivity spread.
    res_aug = B_aug - A_aug @ xi_tik
    res_norm = float(np.linalg.norm(res_aug))

    # ------------------------------------------------------------------
    # Step 3 - Per-variable adaptive encoding
    # ------------------------------------------------------------------
    sigma_alpha = 1.0 / s_r           # sensitivity 1/s_i
    spread = cfg.encoding_margin * (
        np.abs(alpha_tik) + sigma_alpha * res_norm
    )
    R_enc = np.maximum(spread, cfg.min_spread)

    p = int(cfg.bits_per_variable)
    max_int = (1 << p) - 1
    step_sizes = (2.0 * R_enc) / max_int
    alpha_offset = alpha_tik - R_enc

    # ------------------------------------------------------------------
    # Step 4 - QUBO assembly
    # ------------------------------------------------------------------
    A_red = A_aug @ V_r                # (M + N, r); equals U_r @ diag(s_r)

    # D_dec is block-diagonal: row i has step_i * [1, 2, ..., 2^(p-1)]
    # at columns i*p .. (i+1)*p - 1. It's small (r x r*p).
    n_bits = rank * p
    D_dec = np.zeros((rank, n_bits))
    powers = (1 << np.arange(p)).astype(np.float64)
    for i in range(rank):
        D_dec[i, i * p : (i + 1) * p] = step_sizes[i] * powers

    Phi = A_red @ D_dec                         # (M + N, n_bits)
    B_tilde = B_aug - A_red @ alpha_offset      # (M + N,)

    Q = Phi.T @ Phi
    linear = -2.0 * (Phi.T @ B_tilde)
    const = float(B_tilde @ B_tilde)

    return LinearLsqToQuboResult(
        Q=Q,
        linear=linear,
        const=const,
        V_r=V_r,
        alpha_offset=alpha_offset,
        step_sizes=step_sizes,
        bits_per_variable=p,
        alpha_tik=alpha_tik,
        xi_tik=xi_tik,
        singular_values_augmented=s_aug,
        effective_rank=rank,
        lambda_tikhonov=lambda_tik,
        config=cfg,
    )
