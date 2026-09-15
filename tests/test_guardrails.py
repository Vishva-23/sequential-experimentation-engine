from consumer.guardrails import (
    OnlineGuardrails,
    bernoulli_confidence_sequence,
)
from consumer.outcome_join import ResolvedOutcome


def make_outcome(
    transaction_id: str,
    arm: str,
    approved: bool,
    is_fraud: int,
) -> ResolvedOutcome:

    fraud_leakage = int(
        approved and is_fraud == 1
    )

    return ResolvedOutcome(
        transaction_id=transaction_id,
        card_id=f"card_{transaction_id}",
        experiment_id="fraud_threshold_v1",
        arm=arm,
        approved=approved,
        is_fraud=is_fraud,
        fraud_leakage=fraud_leakage,
    )


def test_empty_guardrails_continue():

    guardrails = OnlineGuardrails()

    state = guardrails.snapshot()

    assert state.decision == "CONTINUE"

    assert (
        state.high_value_fraud.status
        == "MONITORING"
    )

    assert (
        state.false_decline.status
        == "MONITORING"
    )


def test_high_value_fraud_is_counted():

    guardrails = OnlineGuardrails(
        high_value_threshold=100.0
    )

    guardrails.register_transaction(
        transaction_id="tx_1",
        amount=150.0,
    )

    guardrails.update(
        make_outcome(
            transaction_id="tx_1",
            arm="TREATMENT",
            approved=True,
            is_fraud=1,
        )
    )

    state = guardrails.snapshot()

    assert (
        state.high_value_fraud.treatment_n
        == 1
    )

    assert (
        state.high_value_fraud.treatment_events
        == 1
    )

    assert (
        state.high_value_fraud.treatment_rate
        == 1.0
    )


def test_low_value_fraud_is_not_high_value_event():

    guardrails = OnlineGuardrails(
        high_value_threshold=100.0
    )

    guardrails.register_transaction(
        transaction_id="tx_1",
        amount=50.0,
    )

    guardrails.update(
        make_outcome(
            transaction_id="tx_1",
            arm="TREATMENT",
            approved=True,
            is_fraud=1,
        )
    )

    state = guardrails.snapshot()

    assert (
        state.high_value_fraud.treatment_n
        == 1
    )

    assert (
        state.high_value_fraud.treatment_events
        == 0
    )


def test_false_decline_is_counted():

    guardrails = OnlineGuardrails()

    guardrails.register_transaction(
        transaction_id="tx_1",
        amount=75.0,
    )

    guardrails.update(
        make_outcome(
            transaction_id="tx_1",
            arm="CONTROL",
            approved=False,
            is_fraud=0,
        )
    )

    state = guardrails.snapshot()

    assert (
        state.false_decline.control_n
        == 1
    )

    assert (
        state.false_decline.control_events
        == 1
    )

    assert (
        state.false_decline.control_rate
        == 1.0
    )


def test_harmful_treatment_can_breach_guardrail():

    guardrails = OnlineGuardrails(
        high_value_threshold=100.0
    )

    for index in range(20):

        control_id = f"control_{index}"

        guardrails.register_transaction(
            transaction_id=control_id,
            amount=150.0,
        )

        guardrails.update(
            make_outcome(
                transaction_id=control_id,
                arm="CONTROL",
                approved=False,
                is_fraud=1,
            )
        )

        treatment_id = f"treatment_{index}"

        guardrails.register_transaction(
            transaction_id=treatment_id,
            amount=150.0,
        )

        guardrails.update(
            make_outcome(
                transaction_id=treatment_id,
                arm="TREATMENT",
                approved=True,
                is_fraud=1,
            )
        )

    state = guardrails.snapshot()

    assert (
        state.high_value_fraud.status
        == "BREACH"
    )

    assert (
        state.high_value_fraud.difference_lower
        > 0.0
    )

    assert state.decision == "STOP_GUARDRAIL"


def test_confidence_sequence_contains_observed_rate():

    lower, upper = (
        bernoulli_confidence_sequence(
            n=1000,
            x=40,
            alpha=0.0125,
        )
    )

    observed = 40 / 1000

    assert lower <= observed <= upper