"""Dense quantum geometric tensor operator for variational optimization.

This module provides a dense QGT implementation compatible with NetKet's
linear operator interface.
"""
from __future__ import annotations

from VMC import config  # noqa: F401 - JAX config must be imported first

import functools
import logging
from typing import Any, Callable

import jax
import jax.numpy as jnp
import netket as nk
from netket.optimizer.linear_operator import LinearOperator
from netket.optimizer.preconditioner import AbstractLinearPreconditioner
from netket.utils.types import PyTree, Scalar

from VMC.utils.vmc_utils import build_dense_jac, get_apply_fun

logger = logging.getLogger(__name__)

__all__ = ["DenseQGTOperator", "MinimalDenseSR"]


@nk.utils.struct.dataclass
class DenseQGTOperator(LinearOperator):
    """Dense quantum geometric tensor S = O^H O with a diagonal shift.

    This operator represents the regularized quantum geometric tensor
    (S + diag_shift * I) where S = O^H O and O is the centered Jacobian.

    Attributes:
        O: Dense Jacobian matrix with shape (n_samples, n_params).
        diag_shift: Regularization added to diagonal.
        holomorphic: Whether the Jacobian was computed in holomorphic mode.
    """

    O: jax.Array | None = None
    diag_shift: float = 1e-2
    holomorphic: bool = nk.utils.struct.field(pytree_node=False, default=True)
    _in_solve: bool = nk.utils.struct.field(pytree_node=False, default=False)
    _params_structure: PyTree = nk.utils.struct.field(pytree_node=False, default=None)

    @staticmethod
    def _to_dense(
        vec: PyTree | jax.Array, *, disable: bool = False
    ) -> tuple[jax.Array, Callable[[jax.Array], PyTree]]:
        """Convert pytree to dense array with unravel function."""
        if disable or hasattr(vec, "ndim"):
            return vec, lambda x: x
        flat, unravel = jax.flatten_util.ravel_pytree(vec)
        return flat, unravel

    @staticmethod
    @jax.jit
    def _mat_vec(vec: jax.Array, O: jax.Array, diag_shift: float) -> jax.Array:
        """Compute (O^H O + diag_shift I) @ vec.

        Supports vec with leading batch dimensions (..., n_params).
        """
        diag = jnp.asarray(diag_shift, dtype=vec.dtype)
        Ov = jnp.einsum("sp,...p->s...", O, vec)
        gram_v = jnp.einsum("ps,s...->...p", O.conj().T, Ov)
        return gram_v + diag * vec

    @functools.partial(jax.jit, inline=True, donate_argnums=(1,))
    def __matmul__(self, v: PyTree) -> PyTree:
        """Matrix-vector product with the QGT."""
        vec, reassemble = self._to_dense(v, disable=self._in_solve)
        result = self._mat_vec(vec, self.O, self.diag_shift)
        return reassemble(result)

    @jax.jit
    def _solve(
        self,
        solve_fun: Callable,
        y: PyTree,
        *,
        x0: PyTree | None = None,
    ) -> tuple[PyTree, Any]:
        """Solve the linear system using the provided solver."""
        y_dense, reassemble = self._to_dense(y)
        x0_dense = None
        if x0 is not None:
            x0_dense, _ = self._to_dense(x0)

        op = self.replace(_in_solve=True)
        out, info = solve_fun(op, y_dense, x0=x0_dense)
        return reassemble(out), info


class MinimalDenseSR(AbstractLinearPreconditioner):
    """Minimal dense SR preconditioner using flattened Jacobian.

    This preconditioner builds a dense QGT from the full Jacobian matrix,
    which is efficient for small to medium-sized parameter counts.

    Attributes:
        diag_shift: Regularization parameter.
        holomorphic: Whether to use holomorphic gradients.
    """

    diag_shift: float
    holomorphic: bool

    def __init__(
        self,
        diag_shift: float = 1e-2,
        solver: Callable = jax.scipy.sparse.linalg.cg,
        holomorphic: bool = True,
    ):
        """Initialize MinimalDenseSR.

        Args:
            diag_shift: Regularization added to QGT diagonal.
            solver: Linear solver (default: conjugate gradient).
            holomorphic: Whether to use holomorphic gradients.
        """
        super().__init__(solver=solver)
        self.diag_shift = diag_shift
        self.holomorphic = holomorphic

    def lhs_constructor(
        self,
        vstate: nk.vqs.VariationalState,
        step: Scalar | None = None,
    ) -> DenseQGTOperator:
        """Construct the dense QGT operator.

        Args:
            vstate: Variational state.
            step: Current optimization step.

        Returns:
            DenseQGTOperator instance.
        """
        samples = vstate.samples
        # Flatten leading dimensions (n_chains, n_samples_per_chain) -> (n_samples,)
        if samples.ndim >= 3:
            samples = samples.reshape(-1, samples.shape[-1])

        apply_fun, params, model_state, _ = get_apply_fun(vstate)

        O_dense = build_dense_jac(
            apply_fun,
            params,
            model_state,
            samples,
            holomorphic=self.holomorphic,
        )

        # Debug logging (guarded)
        if logger.isEnabledFor(logging.DEBUG):
            rank = int(jnp.linalg.matrix_rank(O_dense))
            logger.debug("Dense Jacobian shape: %s, rank: %d", O_dense.shape, rank)

        pars_struct = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params
        )

        return DenseQGTOperator(
            O=O_dense,
            diag_shift=self.diag_shift,
            holomorphic=self.holomorphic,
            _params_structure=pars_struct,
        )
