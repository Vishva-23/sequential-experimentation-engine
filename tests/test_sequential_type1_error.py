import numpy as np

from simulation.validate_sequential import validate_sequential_type1_error


def test_sequential_type1_error_is_controlled():

    alpha = 0.05
    n_experiments = 5_000

    results = validate_sequential_type1_error(
        n_experiments=n_experiments,
        true_rate=0.035,
        max_n_per_arm=22_639,
        alpha=alpha,
        seed=42,
    )

    observed_fpr = results["sequential_fpr"]

    # Monte Carlo standard error at the target alpha
    monte_carlo_se = np.sqrt(
        alpha * (1 - alpha) / n_experiments
    )

    # Allow simulation noise around the theoretical bound
    upper_bound = alpha + 4 * monte_carlo_se

    assert observed_fpr <= upper_bound, (
        f"Sequential Type I error "
        f"{observed_fpr:.3%} exceeded "
        f"allowed bound {upper_bound:.3%}"
    )