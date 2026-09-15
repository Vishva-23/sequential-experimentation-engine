from consumer.experiment_engine import ExperimentEngine
from producer.events import (
    FraudOutcomeEvent,
    TransactionEvent,
)


def make_transaction(
    transaction_id: str,
    card_id: str,
    fraud_score: float = 0.38,
) -> TransactionEvent:

    return TransactionEvent(
        transaction_id=transaction_id,
        card_id=card_id,
        experiment_id="fraud_threshold_v1",
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


def test_operational_metrics_update_before_fraud_outcome():

    engine = ExperimentEngine()

    transaction = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
    )

    state = engine.process_transaction(
        transaction
    )

    # Operational state updates immediately.
    assert state.operational.total_transactions == 1
    assert state.operational.control_n == 1

    # score=0.38 is above the CONTROL threshold 0.35,
    # therefore the transaction is declined.
    assert state.operational.control_approved == 0
    assert state.operational.control_approval_rate == 0.0

    # The delayed fraud label has not arrived yet.
    assert state.resolved.total_resolved == 0


def test_delayed_outcome_does_not_change_operational_counts():

    engine = ExperimentEngine()

    transaction = make_transaction(
        transaction_id="tx_treatment",
        card_id="card_0042",
    )

    state = engine.process_transaction(
        transaction
    )

    # score=0.38 is below the TREATMENT threshold 0.42,
    # therefore the transaction is approved.
    assert state.operational.total_transactions == 1
    assert state.operational.treatment_n == 1
    assert state.operational.treatment_approved == 1
    assert state.operational.treatment_approval_rate == 1.0

    assert state.resolved.total_resolved == 0

    outcome = make_outcome(
        transaction_id="tx_treatment",
        is_fraud=1,
    )

    state = engine.process_outcome(
        outcome
    )

    # Fraud resolution changes delayed primary statistics...
    assert state.resolved.total_resolved == 1

    # ...but must not count the transaction again in
    # immediate operational metrics.
    assert state.operational.total_transactions == 1
    assert state.operational.treatment_n == 1
    assert state.operational.treatment_approved == 1


def test_duplicate_and_second_card_transaction_do_not_inflate_metrics():

    engine = ExperimentEngine()

    first = make_transaction(
        transaction_id="tx_first",
        card_id="card_0042",
    )

    state = engine.process_transaction(
        first
    )

    assert state.total_randomized == 1
    assert state.operational.total_transactions == 1

    # Exact duplicate delivery must be idempotent.
    state = engine.process_transaction(
        first
    )

    assert state.total_randomized == 1
    assert state.operational.total_transactions == 1

    # A different transaction from the same card is
    # ineligible in version 1.
    second = make_transaction(
        transaction_id="tx_second",
        card_id="card_0042",
        fraud_score=0.20,
    )

    state = engine.process_transaction(
        second
    )

    assert state.total_randomized == 1
    assert state.operational.total_transactions == 1
    assert state.operational.treatment_n == 1
    assert state.operational.treatment_approved == 1

    assert (
        "tx_second"
        in engine.ineligible_transaction_ids
    )