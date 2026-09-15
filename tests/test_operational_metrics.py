import pytest

from consumer.operational_metrics import (
    OnlineOperationalMetrics,
)
from consumer.policy import TransactionDecision


def make_decision(
    transaction_id: str,
    arm: str,
    approved: bool,
) -> TransactionDecision:

    return TransactionDecision(
        transaction_id=transaction_id,
        card_id=f"card_{transaction_id}",
        experiment_id="fraud_threshold_v1",
        arm=arm,
        fraud_threshold=(
            0.35 if arm == "CONTROL" else 0.42
        ),
        fraud_score=0.38,
        approved=approved,
        decision=(
            "APPROVE"
            if approved
            else "DECLINE"
        ),
    )


def test_empty_state():

    metrics = OnlineOperationalMetrics()

    state = metrics.snapshot()

    assert state.control_n == 0
    assert state.treatment_n == 0

    assert state.control_approved == 0
    assert state.treatment_approved == 0

    assert state.control_approval_rate == 0.0
    assert state.treatment_approval_rate == 0.0

    assert state.approval_rate_difference == 0.0
    assert state.total_transactions == 0


def test_control_update():

    metrics = OnlineOperationalMetrics()

    metrics.update(
        make_decision(
            transaction_id="c1",
            arm="CONTROL",
            approved=True,
        )
    )

    state = metrics.snapshot()

    assert state.control_n == 1
    assert state.control_approved == 1

    assert state.control_approval_rate == 1.0

    assert state.treatment_n == 0
    assert state.total_transactions == 1


def test_treatment_update():

    metrics = OnlineOperationalMetrics()

    metrics.update(
        make_decision(
            transaction_id="t1",
            arm="TREATMENT",
            approved=False,
        )
    )

    state = metrics.snapshot()

    assert state.treatment_n == 1
    assert state.treatment_approved == 0

    assert state.treatment_approval_rate == 0.0

    assert state.control_n == 0
    assert state.total_transactions == 1


def test_approval_rates_are_computed_correctly():

    metrics = OnlineOperationalMetrics()

    decisions = [
        make_decision(
            transaction_id="c1",
            arm="CONTROL",
            approved=True,
        ),
        make_decision(
            transaction_id="c2",
            arm="CONTROL",
            approved=False,
        ),
        make_decision(
            transaction_id="t1",
            arm="TREATMENT",
            approved=True,
        ),
        make_decision(
            transaction_id="t2",
            arm="TREATMENT",
            approved=True,
        ),
    ]

    for decision in decisions:
        metrics.update(decision)

    state = metrics.snapshot()

    assert state.control_n == 2
    assert state.control_approved == 1

    assert state.treatment_n == 2
    assert state.treatment_approved == 2

    assert state.control_approval_rate == pytest.approx(
        0.5
    )

    assert state.treatment_approval_rate == pytest.approx(
        1.0
    )

    assert state.approval_rate_difference == pytest.approx(
        0.5
    )

    assert state.total_transactions == 4


def test_invalid_arm_is_rejected():

    metrics = OnlineOperationalMetrics()

    bad_decision = TransactionDecision(
        transaction_id="bad",
        card_id="card_bad",
        experiment_id="fraud_threshold_v1",
        arm="INVALID",
        fraud_threshold=0.35,
        fraud_score=0.38,
        approved=True,
        decision="APPROVE",
    )

    with pytest.raises(ValueError):
        metrics.update(
            bad_decision
        )