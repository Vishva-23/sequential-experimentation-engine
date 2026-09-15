import pytest

from consumer.pair_outcomes import CompletedPair
from consumer.sequential_inference import (
    StreamingSequentialInference,
)


def make_pair(
    pair_id: int,
    control_y: int,
    treatment_y: int,
) -> CompletedPair:

    return CompletedPair(
        pair_id=pair_id,
        control_transaction_id=f"tx_c{pair_id}",
        treatment_transaction_id=f"tx_t{pair_id}",
        control_y=control_y,
        treatment_y=treatment_y,
    )


def test_initial_state():

    inference = StreamingSequentialInference()

    snapshot = inference.snapshot()

    assert snapshot.completed_pairs == 0

    assert snapshot.control_leakage == 0
    assert snapshot.treatment_leakage == 0

    assert snapshot.positive_discordants == 0
    assert snapshot.negative_discordants == 0

    assert snapshot.evidence == pytest.approx(1.0)
    assert snapshot.threshold == pytest.approx(20.0)

    assert snapshot.decision == "CONTINUE"


def test_positive_discordance_updates_correctly():

    inference = StreamingSequentialInference()

    result = inference.update(
        make_pair(
            pair_id=1,
            control_y=0,
            treatment_y=1,
        )
    )

    assert result.completed_pairs == 1

    assert result.control_leakage == 0
    assert result.treatment_leakage == 1

    assert result.positive_discordants == 1
    assert result.negative_discordants == 0


def test_negative_discordance_updates_correctly():

    inference = StreamingSequentialInference()

    result = inference.update(
        make_pair(
            pair_id=1,
            control_y=1,
            treatment_y=0,
        )
    )

    assert result.completed_pairs == 1

    assert result.control_leakage == 1
    assert result.treatment_leakage == 0

    assert result.positive_discordants == 0
    assert result.negative_discordants == 1


def test_concordant_pairs_do_not_change_discordance_counts():

    inference = StreamingSequentialInference()

    inference.update(
        make_pair(
            pair_id=1,
            control_y=0,
            treatment_y=0,
        )
    )

    result = inference.update(
        make_pair(
            pair_id=2,
            control_y=1,
            treatment_y=1,
        )
    )

    assert result.completed_pairs == 2

    assert result.control_leakage == 1
    assert result.treatment_leakage == 1

    assert result.positive_discordants == 0
    assert result.negative_discordants == 0

    assert result.evidence == pytest.approx(1.0)


def test_balanced_discordances_do_not_create_strong_evidence():

    inference = StreamingSequentialInference()

    inference.update(
        make_pair(
            pair_id=1,
            control_y=0,
            treatment_y=1,
        )
    )

    result = inference.update(
        make_pair(
            pair_id=2,
            control_y=1,
            treatment_y=0,
        )
    )

    assert result.positive_discordants == 1
    assert result.negative_discordants == 1

    assert result.evidence < result.threshold
    assert result.decision == "CONTINUE"


def test_sustained_positive_direction_crosses_boundary():

    inference = StreamingSequentialInference(
        alpha=0.05,
        prior_a=100.0,
        prior_b=100.0,
    )

    result = None

    for pair_id in range(1, 100):

        result = inference.update(
            make_pair(
                pair_id=pair_id,
                control_y=0,
                treatment_y=1,
            )
        )

        if result.decision == "STOP_REJECT_H0":
            break

    assert result is not None

    assert result.evidence >= result.threshold
    assert result.decision == "STOP_REJECT_H0"

    assert result.positive_discordants > 0
    assert result.negative_discordants == 0


def test_invalid_control_y_is_rejected():

    inference = StreamingSequentialInference()

    bad_pair = make_pair(
        pair_id=1,
        control_y=2,
        treatment_y=0,
    )

    with pytest.raises(ValueError):
        inference.update(bad_pair)


def test_invalid_treatment_y_is_rejected():

    inference = StreamingSequentialInference()

    bad_pair = make_pair(
        pair_id=1,
        control_y=0,
        treatment_y=2,
    )

    with pytest.raises(ValueError):
        inference.update(bad_pair)