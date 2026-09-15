import pytest

from consumer.outcome_join import ResolvedOutcome
from consumer.pair_outcomes import PairOutcomeBuffer
from consumer.pairing import PairAssignment


def make_outcome(
    transaction_id: str,
    arm: str,
    y: int,
) -> ResolvedOutcome:

    return ResolvedOutcome(
        transaction_id=transaction_id,
        card_id=f"card_{transaction_id}",
        experiment_id="fraud_threshold_v1",
        arm=arm,
        approved=bool(y),
        is_fraud=y,
        fraud_leakage=y,
    )


def make_pair() -> PairAssignment:

    return PairAssignment(
        pair_id=1,
        control_transaction_id="tx_c1",
        treatment_transaction_id="tx_t1",
    )


def complete_pair(
    control_y: int,
    treatment_y: int,
):

    buffer = PairOutcomeBuffer()

    buffer.register_pair(
        make_pair()
    )

    first = buffer.process_outcome(
        make_outcome(
            transaction_id="tx_c1",
            arm="CONTROL",
            y=control_y,
        )
    )

    second = buffer.process_outcome(
        make_outcome(
            transaction_id="tx_t1",
            arm="TREATMENT",
            y=treatment_y,
        )
    )

    assert first is None
    assert second is not None

    return second


def test_positive_discordance():

    completed = complete_pair(
        control_y=0,
        treatment_y=1,
    )

    assert completed.control_y == 0
    assert completed.treatment_y == 1

    assert completed.is_discordant is True
    assert completed.direction == "POSITIVE"


def test_negative_discordance():

    completed = complete_pair(
        control_y=1,
        treatment_y=0,
    )

    assert completed.control_y == 1
    assert completed.treatment_y == 0

    assert completed.is_discordant is True
    assert completed.direction == "NEGATIVE"


def test_zero_zero_is_concordant():

    completed = complete_pair(
        control_y=0,
        treatment_y=0,
    )

    assert completed.is_discordant is False
    assert completed.direction == "CONCORDANT"


def test_one_one_is_concordant():

    completed = complete_pair(
        control_y=1,
        treatment_y=1,
    )

    assert completed.is_discordant is False
    assert completed.direction == "CONCORDANT"


def test_treatment_outcome_can_arrive_first():

    buffer = PairOutcomeBuffer()

    buffer.register_pair(
        make_pair()
    )

    first = buffer.process_outcome(
        make_outcome(
            transaction_id="tx_t1",
            arm="TREATMENT",
            y=1,
        )
    )

    assert first is None

    completed = buffer.process_outcome(
        make_outcome(
            transaction_id="tx_c1",
            arm="CONTROL",
            y=0,
        )
    )

    assert completed is not None

    assert completed.control_y == 0
    assert completed.treatment_y == 1

    assert completed.direction == "POSITIVE"


def test_outcome_can_arrive_before_pair_exists():

    buffer = PairOutcomeBuffer()

    first = buffer.process_outcome(
        make_outcome(
            transaction_id="tx_c1",
            arm="CONTROL",
            y=0,
        )
    )

    assert first is None

    pair_result = buffer.register_pair(
        make_pair()
    )

    assert pair_result is None

    completed = buffer.process_outcome(
        make_outcome(
            transaction_id="tx_t1",
            arm="TREATMENT",
            y=1,
        )
    )

    assert completed is not None
    assert completed.direction == "POSITIVE"


def test_duplicate_after_completion_is_ignored():

    buffer = PairOutcomeBuffer()

    pair = make_pair()

    control = make_outcome(
        transaction_id="tx_c1",
        arm="CONTROL",
        y=0,
    )

    treatment = make_outcome(
        transaction_id="tx_t1",
        arm="TREATMENT",
        y=1,
    )

    buffer.register_pair(pair)

    buffer.process_outcome(control)

    completed = buffer.process_outcome(
        treatment
    )

    assert completed is not None

    duplicate = buffer.process_outcome(
        treatment
    )

    assert duplicate is None

    assert "tx_t1" not in buffer.resolved_outcomes


def test_wrong_arm_is_rejected():

    buffer = PairOutcomeBuffer()

    buffer.register_pair(
        make_pair()
    )

    buffer.process_outcome(
        make_outcome(
            transaction_id="tx_c1",
            arm="TREATMENT",
            y=0,
        )
    )

    with pytest.raises(ValueError):
        buffer.process_outcome(
            make_outcome(
                transaction_id="tx_t1",
                arm="TREATMENT",
                y=1,
            )
        )