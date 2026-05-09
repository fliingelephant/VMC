from vmc.operators.local_terms import (
    LocalHamiltonian,
    OneSiteOperator,
    PlaquetteOperator,
    TransitionOperator,
    merge_operators,
    support_span,
)
import jax.numpy as jnp

from vmc.peps.common.block_sparse import build_eval_schedule


def test_eval_schedule_groups_same_row_dr_col_by_dc():
    def eval_span(term: TransitionOperator) -> tuple[int, int]:
        if isinstance(term, OneSiteOperator):
            return 2, 1
        return support_span(term)

    hamiltonian = LocalHamiltonian(
        shape=(3, 3),
        terms=(
            OneSiteOperator(row=0, col=0, op=jnp.eye(2)),
            PlaquetteOperator(row=0, col=0),
        ),
    )
    bucketed_terms, _ = merge_operators(
        (hamiltonian,),
        hamiltonian.shape,
        eval_span=eval_span,
    )

    schedule = build_eval_schedule(bucketed_terms, eval_span)
    row_pass = schedule.rows[0][0]

    assert row_pass.dr == 2
    assert [bucket.dc for bucket in row_pass.columns[0]] == [1, 2]
    assert [len(bucket.terms) for bucket in row_pass.columns[0]] == [1, 1]
    assert schedule.rows[1] == ()
