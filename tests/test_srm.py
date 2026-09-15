from consumer.srm import (
    detect_srm,
    srm_log_evidence,
)


def test_balanced_allocation_is_valid():

    result = detect_srm(
        control_count=5_000,
        treatment_count=5_000,
    )

    assert result.status == "VALID"
    assert result.observed_treatment_fraction == 0.5


def test_small_random_imbalance_is_valid():

    result = detect_srm(
        control_count=5_020,
        treatment_count=4_980,
    )

    assert result.status == "VALID"


def test_severe_imbalance_invalidates_experiment():

    result = detect_srm(
        control_count=7_000,
        treatment_count=3_000,
    )

    assert result.status == "INVALID_EXPERIMENT"
    assert result.log_evidence > 0


def test_zero_assignments_are_valid():

    result = detect_srm(
        control_count=0,
        treatment_count=0,
    )

    assert result.status == "VALID"
    assert result.total_count == 0


def test_negative_counts_are_rejected():

    try:
        srm_log_evidence(
            control_count=-1,
            treatment_count=10,
        )

    except ValueError:
        return

    raise AssertionError(
        "Negative assignment counts should raise ValueError."
    )