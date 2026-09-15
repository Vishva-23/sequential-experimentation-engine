from consumer.outcome_join import OutcomeJoinState
from producer.events import (
    FraudOutcomeEvent,
    TransactionEvent,
)


def make_transaction(
    transaction_id: str,
    card_id: str,
    fraud_score: float,
) -> TransactionEvent:

    return TransactionEvent(
        transaction_id=transaction_id,
        card_id=card_id,
        experiment_id="fraud_threshold_v1",
        timestamp="2026-09-15T16:00:00+00:00",
        amount=50.0,
        fraud_score=fraud_score,
    )


def make_outcome(
    transaction_id: str,
    is_fraud: int,
) -> FraudOutcomeEvent:

    return FraudOutcomeEvent(
        transaction_id=transaction_id,
        timestamp="2026-09-15T17:00:00+00:00",
        is_fraud=is_fraud,
    )


def test_transaction_then_outcome_resolves():

    state = OutcomeJoinState()

    transaction = make_transaction(
        transaction_id="tx_001",
        card_id="card_0042",
        fraud_score=0.38,
    )

    outcome = make_outcome(
        transaction_id="tx_001",
        is_fraud=1,
    )

    first = state.process_transaction(
        transaction
    )

    assert first is None
    assert len(state.pending_decisions) == 1

    resolved = state.process_outcome(
        outcome
    )

    assert resolved is not None
    assert resolved.transaction_id == "tx_001"
    assert resolved.fraud_leakage == 1
    assert len(state.pending_decisions) == 0


def test_outcome_then_transaction_also_resolves():

    state = OutcomeJoinState()

    transaction = make_transaction(
        transaction_id="tx_002",
        card_id="card_0042",
        fraud_score=0.38,
    )

    outcome = make_outcome(
        transaction_id="tx_002",
        is_fraud=1,
    )

    first = state.process_outcome(
        outcome
    )

    assert first is None
    assert len(state.pending_outcomes) == 1

    resolved = state.process_transaction(
        transaction
    )

    assert resolved is not None
    assert resolved.fraud_leakage == 1
    assert len(state.pending_outcomes) == 0


def test_approved_nonfraud_has_zero_leakage():

    state = OutcomeJoinState()

    transaction = make_transaction(
        transaction_id="tx_003",
        card_id="card_0042",
        fraud_score=0.38,
    )

    state.process_transaction(transaction)

    resolved = state.process_outcome(
        make_outcome(
            transaction_id="tx_003",
            is_fraud=0,
        )
    )

    assert resolved is not None
    assert resolved.approved is True
    assert resolved.is_fraud == 0
    assert resolved.fraud_leakage == 0


def test_declined_fraud_has_zero_leakage():

    state = OutcomeJoinState()

    # card_0101 is assigned to CONTROL.
    # Score 0.38 exceeds the control threshold 0.35.
    transaction = make_transaction(
        transaction_id="tx_004",
        card_id="card_0101",
        fraud_score=0.38,
    )

    state.process_transaction(transaction)

    resolved = state.process_outcome(
        make_outcome(
            transaction_id="tx_004",
            is_fraud=1,
        )
    )

    assert resolved is not None
    assert resolved.approved is False
    assert resolved.is_fraud == 1
    assert resolved.fraud_leakage == 0


def test_duplicate_outcome_is_not_counted_twice():

    state = OutcomeJoinState()

    transaction = make_transaction(
        transaction_id="tx_005",
        card_id="card_0042",
        fraud_score=0.38,
    )

    outcome = make_outcome(
        transaction_id="tx_005",
        is_fraud=1,
    )

    state.process_transaction(transaction)

    first = state.process_outcome(outcome)
    second = state.process_outcome(outcome)

    assert first is not None
    assert second is None


def test_duplicate_transaction_is_not_counted_twice():

    state = OutcomeJoinState()

    transaction = make_transaction(
        transaction_id="tx_006",
        card_id="card_0042",
        fraud_score=0.38,
    )

    first = state.process_transaction(transaction)
    second = state.process_transaction(transaction)

    assert first is None
    assert second is None
    assert len(state.pending_decisions) == 1