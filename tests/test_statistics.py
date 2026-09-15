import pytest

from consumer.outcome_join import ResolvedOutcome
from consumer.statistics import OnlineExperimentStats


def make_resolved(
    transaction_id: str,
    arm: str,
    fraud_leakage: int,
) -> ResolvedOutcome:

    return ResolvedOutcome(
        transaction_id=transaction_id,
        card_id=f"card_{transaction_id}",
        experiment_id="fraud_threshold_v1",
        arm=arm,
        approved=bool(fraud_leakage),
        is_fraud=fraud_leakage,
        fraud_leakage=fraud_leakage,
    )


def test_empty_state():

    stats = OnlineExperimentStats()

    snapshot = stats.snapshot()

    assert snapshot.control_n == 0
    assert snapshot.treatment_n == 0
    assert snapshot.control_rate == 0.0
    assert snapshot.treatment_rate == 0.0
    assert snapshot.difference == 0.0
    assert snapshot.total_resolved == 0


def test_control_update():

    stats = OnlineExperimentStats()

    stats.update(
        make_resolved(
            transaction_id="001",
            arm="CONTROL",
            fraud_leakage=1,
        )
    )

    snapshot = stats.snapshot()

    assert snapshot.control_n == 1
    assert snapshot.control_fraud_leakage == 1
    assert snapshot.control_rate == 1.0
    assert snapshot.total_resolved == 1


def test_treatment_update():

    stats = OnlineExperimentStats()

    stats.update(
        make_resolved(
            transaction_id="001",
            arm="TREATMENT",
            fraud_leakage=1,
        )
    )

    snapshot = stats.snapshot()

    assert snapshot.treatment_n == 1
    assert snapshot.treatment_fraud_leakage == 1
    assert snapshot.treatment_rate == 1.0


def test_rates_are_computed_correctly():

    stats = OnlineExperimentStats()

    observations = [
        ("001", "CONTROL", 0),
        ("002", "CONTROL", 1),
        ("003", "TREATMENT", 1),
        ("004", "TREATMENT", 1),
    ]

    for transaction_id, arm, leakage in observations:

        stats.update(
            make_resolved(
                transaction_id=transaction_id,
                arm=arm,
                fraud_leakage=leakage,
            )
        )

    snapshot = stats.snapshot()

    assert snapshot.control_n == 2
    assert snapshot.control_fraud_leakage == 1

    assert snapshot.treatment_n == 2
    assert snapshot.treatment_fraud_leakage == 2

    assert snapshot.control_rate == 0.5
    assert snapshot.treatment_rate == 1.0
    assert snapshot.difference == 0.5
    assert snapshot.total_resolved == 4


def test_invalid_arm_is_rejected():

    stats = OnlineExperimentStats()

    outcome = make_resolved(
        transaction_id="001",
        arm="INVALID",
        fraud_leakage=0,
    )

    with pytest.raises(ValueError):
        stats.update(outcome)


def test_invalid_fraud_leakage_is_rejected():

    stats = OnlineExperimentStats()

    outcome = ResolvedOutcome(
        transaction_id="tx_bad",
        card_id="card_bad",
        experiment_id="fraud_threshold_v1",
        arm="CONTROL",
        approved=True,
        is_fraud=1,
        fraud_leakage=2,
    )

    with pytest.raises(ValueError):
        stats.update(outcome)