"""NetKet compatibility layer for QGT."""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import netket as nk
from netket.optimizer.linear_operator import LinearOperator
from netket.optimizer.preconditioner import AbstractLinearPreconditioner
from netket.utils.types import PyTree

from VMC import config  # noqa: F401
from VMC.qgt.jacobian import Jacobian
from VMC.qgt.qgt import QGT, ParameterSpace
from VMC.utils.vmc_utils import build_dense_jac, flatten_samples, get_apply_fun

__all__ = ["QGTOperator", "DenseSR"]


@nk.utils.struct.dataclass
class QGTOperator(LinearOperator):
    """NetKet LinearOperator wrapping our QGT."""

    diag_shift: float = 0.0
    _qgt: QGT | None = nk.utils.struct.field(pytree_node=True, default=None)
    _params_structure: PyTree = nk.utils.struct.field(pytree_node=False, default=None)

    def __matmul__(self, v: PyTree) -> PyTree:
        flat, unravel = jax.flatten_util.ravel_pytree(v)
        return unravel(self._qgt @ flat + self.diag_shift * flat)

    def _solve(
        self, solve_fun: Callable, y: PyTree, *, x0: PyTree | None = None
    ) -> tuple[PyTree, Any]:
        y_flat, unravel = jax.flatten_util.ravel_pytree(y)
        out, info = solve_fun(self, y_flat, x0=x0)
        return unravel(out), info

    def to_dense(self) -> jax.Array:
        S = self._qgt.to_dense()
        return S + self.diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)


class DenseSR(AbstractLinearPreconditioner):
    """NetKet SR preconditioner using our QGT."""

    diag_shift: float
    holomorphic: bool

    def __init__(
        self,
        diag_shift: float = 1e-2,
        solver: Callable = jax.scipy.sparse.linalg.cg,
        holomorphic: bool = True,
    ):
        super().__init__(solver=solver)
        self.diag_shift = diag_shift
        self.holomorphic = holomorphic

    def lhs_constructor(
        self, vstate: nk.vqs.VariationalState, step=None
    ) -> QGTOperator:
        samples = flatten_samples(vstate.samples)
        apply_fun, params, model_state, _ = get_apply_fun(vstate)
        O = build_dense_jac(apply_fun, params, model_state, samples, holomorphic=self.holomorphic)
        params_struct = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params
        )
        qgt = QGT(Jacobian(O), space=ParameterSpace())
        return QGTOperator(
            _qgt=qgt,
            diag_shift=self.diag_shift,
            _params_structure=params_struct,
        )
