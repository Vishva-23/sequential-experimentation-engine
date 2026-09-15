from simulation.confidence_sequence import (
    bernoulli_confidence_sequence,
    difference_confidence_sequence,
)


def test_observed_rate_is_inside_confidence_sequence():

    successes = 350
    n = 10_000

    observed_rate = successes / n

    lower, upper = bernoulli_confidence_sequence(
        successes=successes,
        n=n,
    )

    assert lower <= observed_rate <= upper


def test_difference_estimate_is_inside_confidence_sequence():

    result = difference_confidence_sequence(
        control_successes=350,
        control_n=10_000,
        treatment_successes=400,
        treatment_n=10_000,
    )

    assert (
        result.difference_lower
        <= result.difference
        <= result.difference_upper
    )


def test_confidence_sequence_narrows_with_more_data():

    small_lower, small_upper = (
        bernoulli_confidence_sequence(
            successes=35,
            n=1_000,
        )
    )

    large_lower, large_upper = (
        bernoulli_confidence_sequence(
            successes=350,
            n=10_000,
        )
    )

    small_width = (
        small_upper - small_lower
    )

    large_width = (
        large_upper - large_lower
    )

    assert large_width < small_width