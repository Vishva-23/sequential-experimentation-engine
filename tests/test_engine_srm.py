from consumer.experiment_engine import ExperimentEngine
from consumer.pair_outcomes import CompletedPair
from producer.events import (
    FraudOutcomeEvent,
    TransactionEvent,
)


def make_transaction(
    transaction_id: str,
    card_id: str,
) -> TransactionEvent:

    return TransactionEvent(
        transaction_id=transaction_id,
        card_id=card_id,
        experiment_id="fraud_threshold_v1",
        timestamp="2026-09-15T20:00:00+00:00",
        amount=75.00,
        fraud_score=0.38,
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


def test_initial_srm_state_is_valid():

    engine = ExperimentEngine()

    state = engine.snapshot()

    assert state.srm.control_count == 0
    assert state.srm.treatment_count == 0
    assert state.srm.total_count == 0

    assert state.srm.threshold == 1000.0
    assert state.srm.status == "VALID"

    assert state.decision == "CONTINUE"


def test_srm_uses_randomized_not_resolved_counts():

    engine = ExperimentEngine()

    transaction = make_transaction(
        transaction_id="tx_control",
        card_id="card_0101",
    )

    state = engine.process_transaction(
        transaction
    )

    # The card has already entered randomization.
    assert state.total_randomized == 1
    assert state.srm.total_count == 1

    # But its delayed fraud outcome has not arrived.
    assert state.resolved.total_resolved == 0

    outcome = make_outcome(
        transaction_id="tx_control",
        is_fraud=1,
    )

    state = engine.process_outcome(
        outcome
    )

    # Resolving the fraud outcome must not change
    # the SRM sample size.
    assert state.srm.total_count == 1

    assert state.resolved.total_resolved == 1


def test_severe_srm_sets_invalid_experiment():

    engine = ExperimentEngine()

    # Simulate the randomized-card counts of a badly
    # imbalanced experiment.
    engine.randomized_control = 7_000
    engine.randomized_treatment = 3_000

    state = engine.snapshot()

    assert state.srm.control_count == 7_000
    assert state.srm.treatment_count == 3_000

    assert (
        state.srm.observed_treatment_fraction
        == 0.30
    )

    assert state.srm.status == "INVALID_EXPERIMENT"

    assert state.decision == "INVALID_EXPERIMENT"


def test_srm_invalidity_overrides_sequential_stop():

    engine = ExperimentEngine()

    # First create overwhelming primary-metric
    # evidence in one direction.
    pair_id = 1

    while (
        engine.inference.snapshot().decision
        != "STOP_REJECT_H0"
    ):

        pair = CompletedPair(
            pair_id=pair_id,
            control_transaction_id=(
                f"tx_c{pair_id}"
            ),
            treatment_transaction_id=(
                f"tx_t{pair_id}"
            ),
            control_y=0,
            treatment_y=1,
        )

        engine.inference.update(
            pair
        )

        pair_id += 1

        if pair_id > 1_000:
            raise AssertionError(
                "Sequential boundary was not crossed."
            )

    inference_state = (
        engine.inference.snapshot()
    )

    assert (
        inference_state.decision
        == "STOP_REJECT_H0"
    )

    assert (
        inference_state.evidence
        >= inference_state.threshold
    )

    # Now represent a broken randomization process.
    engine.randomized_control = 7_000
    engine.randomized_treatment = 3_000

    state = engine.snapshot()

    # The statistical test itself still contains
    # strong evidence.
    assert (
        state.inference.decision
        == "STOP_REJECT_H0"
    )

    # But the experiment-level decision must refuse
    # to act on that evidence because SRM indicates
    # invalid randomization.
    assert state.srm.status == "INVALID_EXPERIMENT"

    assert state.decision == "INVALID_EXPERIMENT"