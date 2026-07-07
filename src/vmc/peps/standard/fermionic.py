"""Fermionic standard-PEPS MC kernels in the graded sampled basis.

The graded model is the ordinary :class:`~vmc.peps.standard.model.PEPS` plus
:class:`~vmc.peps.grading.Grading` metadata.  After sampling, statistics
reduces to the elementwise assembly rule (``compat._decorate``: even mask x
right-leg gate sign) and scalar signs:

- sampling: adjacent-label exchanges within the fixed sector; proposals are
  symmetric and acceptance reads ``|amp|^2`` ratios only, so the sampler is
  statistics-blind.  Horizontal sweeps run against the cached (stale-gate)
  bottom environments through per-column interface exponents ``delta`` --
  the exact re-gauge of every downstream gate flipped by moves above --
  and vertical pair sweeps are gate-local,
- estimates: fermionic two-site terms decorate the moved-endpoint leg and
  (horizontal only) multiply the Jordan-Wigner string sign.

All environment reuse of the bosonic kernels is preserved: cached bottom
envs feed the transition, top envs hand over to the estimate, and every
window uses left-prefix updates against precomputed right environments.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vmc.operators.local_terms import (
    FermionicHorizontalTwoSiteOperator,
    FermionicVerticalTwoSiteOperator,
    merge_operators,
)
from vmc.peps.common.contraction import (
    _apply_mpo_from_below,
    _build_row_mpo,
    _compute_right_envs,
    _contract_1row_1col,
    _contract_bottom,
)
from vmc.peps.common.energy import (
    RowEnvs,
    TwoRowEnvs,
    _compute_right_envs_2row,
    _estimate_sweep,
    _eval_term,
    _update_left_env_1row,
    _update_left_env_2row,
)
from vmc.peps.common.kernels import (
    Cache,
    Context,
    LocalEstimates,
    _assemble_log_derivatives,
    _broadcast_coeffs,
)
from vmc.peps.grading import FermionSigns, _grading_statics, column_prefix_parities
from vmc.peps.standard.compat import _decorate
from vmc.peps.standard.model import PEPS
from vmc.utils.utils import _metropolis_hastings_accept

__all__ = ["build_fermionic_kernels"]


@_eval_term.dispatch
def _eval_term(
    term: FermionicHorizontalTwoSiteOperator,
    envs: RowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    del tensors
    signs = envs.config
    amps = jnp.einsum(
        "ace,aub,edf,pudcr,qvwrx,bvg,fwi,gxi->pq",
        envs.left_env,
        envs.top_env[col],
        envs.bottom_env[col],
        envs.row_tensors[col],
        envs.row_tensors[col + 1]
        * signs.down_flip[row][col + 1][None, None, :, None, None],
        envs.top_env[col + 1],
        envs.bottom_env[col + 1],
        envs.right_envs[col + 1],
        optimize=[(0, 1), (1, 6), (0, 5), (1, 3), (1, 2), (1, 2), (0, 1)],
    )
    # Jordan-Wigner string of the column-major mode order (suffix[c] +
    # prefix[c+1]) plus the scalar of the downstream re-gauge paired with
    # the down-leg flip above (suffix[c+1]).
    exponent = (
        signs.suffix[row, col] + signs.prefix[row, col + 1] + signs.suffix[row, col + 1]
    )
    s0, s1 = spins[row, col], spins[row, col + 1]
    return (1.0 - 2.0 * (exponent % 2)) * jnp.dot(
        term.op[:, s0 * phys_dim + s1], amps.reshape(-1)
    )


@_eval_term.dispatch
def _eval_term(
    term: FermionicVerticalTwoSiteOperator,
    envs: TwoRowEnvs,
    tensors: Any,
    row: int,
    col: int,
    spins: jax.Array,
    phys_dim: int,
) -> jax.Array:
    del tensors
    amps = jnp.einsum(
        "almg,aub,puvlr,qvwmn,gwf,brnf->pq",
        envs.left_env,
        envs.top_env[col],
        envs.row_tensors[col],
        envs.row_tensors_next[col]
        * envs.config.right_flip[row + 1][col][None, None, None, None, :],
        envs.bottom_env_next[col],
        envs.right_envs[col],
        optimize=[(0, 1), (2, 3), (0, 2), (1, 2), (0, 1)],
    )
    s0, s1 = spins[row, col], spins[row + 1, col]
    return jnp.dot(term.op[:, s0 * phys_dim + s1], amps.reshape(-1))


def build_fermionic_kernels(
    model: PEPS,
    operator: object,
    *,
    full_gradient: bool = False,
    observables: tuple = (),
) -> tuple[Any, Any, Any]:
    """Build init_cache/transition/estimate kernels for a graded PEPS."""
    shape = model.shape
    n_rows, n_cols = shape
    n_sites = n_rows * n_cols
    strategy = model.strategy
    grading = model.grading
    phys_parity = jnp.asarray(grading.phys_parity)
    params_per_site = jnp.asarray(model.params_per_site, dtype=jnp.int32)
    total_active_params = int(sum(model.params_per_site))

    masks, right_par, down_par = _grading_statics(grading, model.tensors)
    down_flip = [[1.0 - 2.0 * par for par in row] for row in down_par]
    right_flip = [[1.0 - 2.0 * par for par in row] for row in right_par]

    all_operators = (operator,) + observables
    terms, coeff_structure = merge_operators(
        all_operators, shape, eval_span=type(model).eval_span
    )
    static_coeffs = coeff_structure.static_coeffs()

    def bottom_envs_from_mpos(mpos: list, first_row: int = 0) -> tuple:
        dtype = mpos[n_rows - 1][0].dtype
        envs = [None] * n_rows
        env = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))
        for row in range(n_rows - 1, first_row - 1, -1):
            envs[row] = env
            if row > first_row:
                env = _apply_mpo_from_below(env, tuple(mpos[row]), strategy)
        return tuple(envs)

    def init_cache(
        tensors: Any,
        samples: jax.Array,
        t: float | jax.Array | None = None,
    ) -> Cache:
        samples_flat = samples.reshape(-1, n_sites)

        def build_one(sample: jax.Array) -> tuple:
            spins = sample.reshape(shape)
            decorated = _decorate(
                tensors, column_prefix_parities(phys_parity[spins]), masks, right_par
            )
            return bottom_envs_from_mpos(
                [_build_row_mpo(decorated, spins[r], r, n_cols) for r in range(n_rows)]
            )

        return Cache(
            bottom_envs=jax.vmap(build_one)(samples_flat),
            coeffs=_broadcast_coeffs(
                coeff_structure.dynamic_coeffs(t),
                samples_flat.shape[0],
            ),
        )

    def transition(
        tensors: Any,
        sample: jax.Array,
        key: jax.Array,
        cache: Cache,
    ) -> tuple[jax.Array, jax.Array, Context]:
        spins = sample.reshape(shape)
        dtype = jnp.asarray(tensors[0][0]).dtype
        tm = [
            [jnp.asarray(tensors[r][c]) * masks[r][c] for c in range(n_cols)]
            for r in range(n_rows)
        ]
        parities = phys_parity[spins]
        trivial = tuple(jnp.ones((1, 1, 1), dtype=dtype) for _ in range(n_cols))

        # Phase A: horizontal exchanges row by row against cached bottom
        # envs.  ``delta[c]`` is the interface exponent re-gauging every
        # stale gate below the current row into a down-leg sign; it flips
        # at ``c + 1`` when a move at bond (c, c+1) transports parity.  The
        # maintained ``amp`` may drift from the current re-gauge sign
        # convention across rows, but its magnitude stays the true one and
        # acceptance reads magnitudes only.
        top_env = trivial
        top_envs_cache = [None] * n_rows
        mpos = [None] * n_rows
        delta = jnp.zeros((n_cols,), dtype=parities.dtype)
        pi_row = jnp.zeros((n_cols,), dtype=parities.dtype)
        for row in range(n_rows):
            top_envs_cache[row] = top_env
            bottom_env = cache.bottom_envs[row]
            dec = [
                tm[row][c] * (1.0 - 2.0 * pi_row[c] * right_par[row][c])
                for c in range(n_cols)
            ]
            mpo = [
                jnp.transpose(dec[c][spins[row, c]], (2, 3, 0, 1))
                * (1.0 - 2.0 * delta[c] * down_par[row][c])
                for c in range(n_cols)
            ]
            right_envs = _compute_right_envs(top_env, tuple(mpo), bottom_env, dtype)
            left_env = jnp.ones((1, 1, 1), dtype=dtype)
            if row == 0:
                amp = jnp.einsum(
                    "ace,aub,cduv,evf,bdf->",
                    left_env,
                    top_env[0],
                    mpo[0],
                    bottom_env[0],
                    right_envs[0],
                    optimize=[(0, 1), (1, 2), (1, 2), (0, 1)],
                )
            for c in range(n_cols - 1):
                s0, s1 = spins[row, c], spins[row, c + 1]
                flip = (parities[row, c] + parities[row, c + 1]) % 2
                prop_c = jnp.transpose(dec[c][s1], (2, 3, 0, 1)) * (
                    1.0 - 2.0 * delta[c] * down_par[row][c]
                )
                prop_c1 = jnp.transpose(dec[c + 1][s0], (2, 3, 0, 1)) * (
                    1.0 - 2.0 * ((delta[c + 1] + flip) % 2) * down_par[row][c + 1]
                )
                prefix_cur = _update_left_env_1row(
                    left_env, top_env[c], mpo[c], bottom_env[c]
                )
                prefix_prop = _update_left_env_1row(
                    left_env, top_env[c], prop_c, bottom_env[c]
                )
                amp_prop = _contract_1row_1col(
                    prefix_prop,
                    top_env[c + 1],
                    prop_c1,
                    bottom_env[c + 1],
                    right_envs[c + 1],
                )
                key, accept = _metropolis_hastings_accept(
                    key,
                    jnp.abs(amp) ** 2,
                    jnp.abs(amp_prop) ** 2,
                )
                spins = jnp.where(
                    accept,
                    spins.at[row, c].set(s1).at[row, c + 1].set(s0),
                    spins,
                )
                parities = phys_parity[spins]
                delta = jnp.where(
                    accept, delta.at[c + 1].set((delta[c + 1] + flip) % 2), delta
                )
                mpo[c] = jnp.where(accept, prop_c, mpo[c])
                mpo[c + 1] = jnp.where(accept, prop_c1, mpo[c + 1])
                left_env = jnp.where(accept, prefix_prop, prefix_cur)
                amp = jnp.where(accept, amp_prop, amp)
            # Strip the interface exponents before folding into the true top
            # boundary (the correction masks square to one); the stripped rows
            # are the exact post-accept row MPOs, reused by Phase B.
            mpos[row] = [
                mpo[c] * (1.0 - 2.0 * delta[c] * down_par[row][c])
                for c in range(n_cols)
            ]
            if row + 1 < n_rows:
                top_env = strategy.apply(top_env, tuple(mpos[row]))
            pi_row = (pi_row + parities[row]) % 2

        # Phase B: vertical exchanges on row pairs against rebuilt bottom
        # envs (Phase A's stripped rows are the exact post-accept MPOs);
        # every gate flip is inside the two-row window.
        if n_rows > 1:
            bottom_envs = bottom_envs_from_mpos(mpos, first_row=1)
            top_env = trivial
            pi_top = jnp.zeros((n_cols,), dtype=parities.dtype)
            for r in range(n_rows - 1):
                top_envs_cache[r] = top_env
                bottom_env = bottom_envs[r + 1]
                pi_bot = (pi_top + parities[r]) % 2
                right_envs = _compute_right_envs_2row(
                    top_env, tuple(mpos[r]), tuple(mpos[r + 1]), bottom_env, dtype
                )
                left_env = jnp.ones((1, 1, 1, 1), dtype=dtype)
                for c in range(n_cols):
                    s0, s1 = spins[r, c], spins[r + 1, c]
                    flip = (parities[r, c] + parities[r + 1, c]) % 2
                    prop0 = jnp.transpose(
                        tm[r][c][s1] * (1.0 - 2.0 * pi_top[c] * right_par[r][c]),
                        (2, 3, 0, 1),
                    )
                    prop1 = jnp.transpose(
                        tm[r + 1][c][s0]
                        * (1.0 - 2.0 * ((pi_bot[c] + flip) % 2) * right_par[r + 1][c]),
                        (2, 3, 0, 1),
                    )
                    prefix_cur = _update_left_env_2row(
                        left_env, top_env[c], mpos[r][c], mpos[r + 1][c], bottom_env[c]
                    )
                    prefix_prop = _update_left_env_2row(
                        left_env, top_env[c], prop0, prop1, bottom_env[c]
                    )
                    amp_prop = jnp.einsum("bryf,bryf->", prefix_prop, right_envs[c])
                    key, accept = _metropolis_hastings_accept(
                        key,
                        jnp.abs(amp) ** 2,
                        jnp.abs(amp_prop) ** 2,
                    )
                    spins = jnp.where(
                        accept,
                        spins.at[r, c].set(s1).at[r + 1, c].set(s0),
                        spins,
                    )
                    parities = phys_parity[spins]
                    mpos[r][c] = jnp.where(accept, prop0, mpos[r][c])
                    mpos[r + 1][c] = jnp.where(accept, prop1, mpos[r + 1][c])
                    left_env = jnp.where(accept, prefix_prop, prefix_cur)
                    amp = jnp.where(accept, amp_prop, amp)
                top_env = strategy.apply(top_env, tuple(mpos[r]))
                pi_top = (pi_top + parities[r]) % 2
            top_envs_cache[n_rows - 1] = top_env

        top_env = strategy.apply(top_env, tuple(mpos[n_rows - 1]))
        return (
            spins.reshape(-1),
            key,
            Context(
                amp=_contract_bottom(top_env),
                top_envs=tuple(top_envs_cache),
                coeffs=cache.coeffs,
            ),
        )

    def estimate(
        tensors: Any,
        config_state_next: jax.Array,
        context: Context,
    ) -> tuple[Cache, LocalEstimates]:
        spins = config_state_next.reshape(shape)
        parities = phys_parity[spins]
        prefix = column_prefix_parities(parities)
        suffix = (jnp.sum(parities, axis=0) + prefix + parities) % 2
        decorated = _decorate(tensors, prefix, masks, right_par)

        def build_row_mpo(tensors_: Any, sample_: jax.Array, row: int) -> tuple:
            return _build_row_mpo(tensors_, sample_[row], row, n_cols)

        env_grads, local_estimate, bottom_envs_next = _estimate_sweep(
            decorated,
            spins,
            context.amp,
            context.top_envs,
            strategy=strategy,
            terms=terms,
            build_row_mpo=build_row_mpo,
            env_config=FermionSigns(prefix, suffix, down_flip, right_flip),
            coeffs=static_coeffs if context.coeffs is None else context.coeffs,
            collect_grads=True,
        )
        for r in range(n_rows):
            for c in range(n_cols):
                env_grads[r][c] = env_grads[r][c] * (
                    masks[r][c][spins[r, c]]
                    * (1.0 - 2.0 * prefix[r, c] * right_par[r][c])
                )
        local_log_derivatives, active_slice_indices = _assemble_log_derivatives(
            tensors,
            params_per_site,
            total_active_params,
            shape,
            env_grads,
            config_state_next,
            context.amp,
            full_gradient=full_gradient,
        )
        return Cache(
            bottom_envs=tuple(bottom_envs_next),
            coeffs=context.coeffs,
        ), LocalEstimates(
            local_log_derivatives=local_log_derivatives,
            local_estimate=local_estimate,
            active_slice_indices=active_slice_indices,
            amp=context.amp,
        )

    return init_cache, transition, estimate
