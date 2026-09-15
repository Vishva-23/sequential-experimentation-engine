import pytest

from consumer.experiment_engine import ExperimentEngine
from producer.events import (
    FraudOutcomeEvent,
    TransactionEvent,
)


def make_transaction(
    transaction_id: str,
    card_id: str,
    fraud_score: float = 0.38,
    experiment_id: str = "fraud_threshold_v1",
) -> TransactionEvent:

    return TransactionEvent(
        transaction_id=transaction_id,
        card_id=card_id,
        experiment_id=experiment_id,
        timestamp="2026-09-15T20:00:00+00:00",
        amount=75.00,
        fraud_score=fraud_score,
    )


def make_outcome(
    transaction_id: str,
    is_fraud: int,
) -> FraudOutcomeEvent:

    return FraudOutcomeEvent(
        transaction_id=transaction_id,
        timestamp="2026-09-15T20:30:00+00:00",
        is_fraud=is_fraud,
    )


def test_transaction_is_randomized_before_outcome():

    engine = ExperimentEngine()

    transaction = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
    )

    state = engine.process_transaction(
        transaction
    )

    assert state.randomized_control == 1
    assert state.randomized_treatment == 0
    assert state.total_randomized == 1

    assert state.resolved.total_resolved == 0
    assert state.inference.completed_pairs == 0


def test_end_to_end_positive_discordance():

    engine = ExperimentEngine()

    control = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
        fraud_score=0.38,
    )

    treatment = make_transaction(
        transaction_id="tx_treatment",
        card_id="card_0042",
        fraud_score=0.38,
    )

    engine.process_transaction(control)
    engine.process_transaction(treatment)

    engine.process_outcome(
        make_outcome(
            transaction_id="tx_control",
            is_fraud=1,
        )
    )

    state = engine.process_outcome(
        make_outcome(
            transaction_id="tx_treatment",
            is_fraud=1,
        )
    )

    assert state.randomized_control == 1
    assert state.randomized_treatment == 1

    assert state.resolved.control_n == 1
    assert state.resolved.treatment_n == 1

    # CONTROL:
    # score 0.38 > threshold 0.35
    # so it is declined and Y = 0.
    assert (
        state.resolved.control_fraud_leakage
        == 0
    )

    # TREATMENT:
    # score 0.38 < threshold 0.42
    # so it is approved and fraudulent,
    # therefore Y = 1.
    assert (
        state.resolved.treatment_fraud_leakage
        == 1
    )

    assert state.inference.completed_pairs == 1

    assert (
        state.inference.positive_discordants
        == 1
    )

    assert (
        state.inference.negative_discordants
        == 0
    )

    assert state.inference.decision == "CONTINUE"


def test_outcome_can_arrive_before_transaction():

    engine = ExperimentEngine()

    outcome = make_outcome(
        transaction_id="tx_control",
        is_fraud=1,
    )

    state = engine.process_outcome(
        outcome
    )

    assert state.total_randomized == 0
    assert state.resolved.total_resolved == 0

    transaction = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
        fraud_score=0.38,
    )

    state = engine.process_transaction(
        transaction
    )

    assert state.total_randomized == 1
    assert state.randomized_control == 1

    assert state.resolved.total_resolved == 1

    # CONTROL declines score 0.38,
    # therefore fraud leakage is zero
    # even though the latent label is fraud.
    assert (
        state.resolved.control_fraud_leakage
        == 0
    )


def test_treatment_outcome_can_resolve_first():

    engine = ExperimentEngine()

    control = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
    )

    treatment = make_transaction(
        transaction_id="tx_treatment",
        card_id="card_0042",
    )

    engine.process_transaction(control)
    engine.process_transaction(treatment)

    first_state = engine.process_outcome(
        make_outcome(
            transaction_id="tx_treatment",
            is_fraud=1,
        )
    )

    assert first_state.resolved.total_resolved == 1
    assert first_state.inference.completed_pairs == 0

    final_state = engine.process_outcome(
        make_outcome(
            transaction_id="tx_control",
            is_fraud=1,
        )
    )

    assert final_state.resolved.total_resolved == 2
    assert final_state.inference.completed_pairs == 1

    assert (
        final_state.inference.positive_discordants
        == 1
    )


def test_duplicate_transaction_does_not_randomize_twice():

    engine = ExperimentEngine()

    transaction = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
    )

    first = engine.process_transaction(
        transaction
    )

    second = engine.process_transaction(
        transaction
    )

    assert first.total_randomized == 1
    assert second.total_randomized == 1

    assert second.randomized_control == 1
    assert second.randomized_treatment == 0


def test_second_transaction_from_same_card_is_ineligible():

    engine = ExperimentEngine()

    first = make_transaction(
        transaction_id="tx_first",
        card_id="card_0101",
    )

    second = make_transaction(
        transaction_id="tx_second",
        card_id="card_0101",
    )

    engine.process_transaction(first)

    state = engine.process_transaction(
        second
    )

    assert state.total_randomized == 1

    assert "tx_second" in (
        engine.ineligible_transaction_ids
    )

    # An outcome for the second transaction
    # must not enter experiment statistics.
    state = engine.process_outcome(
        make_outcome(
            transaction_id="tx_second",
            is_fraud=1,
        )
    )

    assert state.resolved.total_resolved == 0
    assert state.inference.completed_pairs == 0


def test_conflicting_duplicate_transaction_is_rejected():

    engine = ExperimentEngine()

    original = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
        fraud_score=0.20,
    )

    conflicting = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
        fraud_score=0.30,
    )

    engine.process_transaction(
        original
    )

    with pytest.raises(ValueError):
        engine.process_transaction(
            conflicting
        )


def test_wrong_experiment_is_rejected():

    engine = ExperimentEngine(
        experiment_id="fraud_threshold_v1"
    )

    transaction = make_transaction(
        transaction_id="tx_wrong",
        card_id="card_0101",
        experiment_id="different_experiment",
    )

    with pytest.raises(ValueError):
        engine.process_transaction(
            transaction
        )