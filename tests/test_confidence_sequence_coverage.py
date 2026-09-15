import numpy as np

from simulation.validate_confidence_sequence import (
    validate_confidence_sequence_coverage,
)


def test_time_uniform_confidence_sequence_coverage():

    alpha = 0.05
    n_experiments = 500

    results = validate_confidence_sequence_coverage(
        n_experiments=n_experiments,
        true_rate=0.035,
        max_n_per_arm=5_000,
        overall_alpha=alpha,
        batch_size=50,
        seed=42,
    )

    observed_failure_rate = (
        results["joint_failure_rate"]
    )

    monte_carlo_se = np.sqrt(
        alpha
        * (1 - alpha)
        / n_experiments
    )

    upper_bound = (
        alpha + 4 * monte_carlo_se
    )

    assert observed_failure_rate <= upper_bound, (
        f"Time-uniform CS failure rate "
        f"{observed_failure_rate:.3%} exceeded "
        f"allowed Monte Carlo bound "
        f"{upper_bound:.3%}"
    )