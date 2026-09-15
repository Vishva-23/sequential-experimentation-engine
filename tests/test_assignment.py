from consumer.assignment import (
    assign_card,
    assignment_value,
)


def test_same_card_always_gets_same_assignment():

    first = assign_card(
        experiment_id="fraud_threshold_v1",
        card_id="card_0042",
    )

    second = assign_card(
        experiment_id="fraud_threshold_v1",
        card_id="card_0042",
    )

    assert first.arm == second.arm
    assert (
        first.fraud_threshold
        == second.fraud_threshold
    )


def test_assignment_uses_correct_threshold():

    for i in range(1_000):

        result = assign_card(
            experiment_id="fraud_threshold_v1",
            card_id=f"card_{i:06d}",
        )

        if result.arm == "CONTROL":
            assert result.fraud_threshold == 0.35

        elif result.arm == "TREATMENT":
            assert result.fraud_threshold == 0.42

        else:
            raise AssertionError(
                f"Unexpected arm: {result.arm}"
            )


def test_assignment_is_approximately_balanced():

    n_cards = 10_000

    treatment_count = 0

    for i in range(n_cards):

        result = assign_card(
            experiment_id="fraud_threshold_v1",
            card_id=f"card_{i:06d}",
        )

        if result.arm == "TREATMENT":
            treatment_count += 1

    treatment_fraction = (
        treatment_count / n_cards
    )

    assert 0.47 <= treatment_fraction <= 0.53, (
        f"Treatment allocation was "
        f"{treatment_fraction:.2%}"
    )


def test_assignment_value_is_between_zero_and_one():

    value = assignment_value(
        experiment_id="fraud_threshold_v1",
        card_id="card_0042",
    )

    assert 0 <= value < 1