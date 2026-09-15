import numpy as np

from simulation.fixed_horizon import two_proportion_z_test


def test_identical_groups_do_not_reject():
    result = two_proportion_z_test(
        control_successes=350,
        control_n=10_000,
        treatment_successes=350,
        treatment_n=10_000,
    )

    assert result.p_value == 1.0


def test_clear_difference_is_detected():
    result = two_proportion_z_test(
        control_successes=350,
        control_n=10_000,
        treatment_successes=500,
        treatment_n=10_000,
    )

    assert result.p_value < 0.05


def test_fixed_horizon_type1_error():
    rng = np.random.default_rng(42)

    n_experiments = 5_000
    true_rate = 0.035
    n_per_arm = 22_639
    alpha = 0.05

    false_positives = 0

    for _ in range(n_experiments):

        # A/A experiment: both groups have the same true fraud rate
        control_frauds = rng.binomial(
            n=n_per_arm,
            p=true_rate,
        )

        treatment_frauds = rng.binomial(
            n=n_per_arm,
            p=true_rate,
        )

        result = two_proportion_z_test(
            control_successes=control_frauds,
            control_n=n_per_arm,
            treatment_successes=treatment_frauds,
            treatment_n=n_per_arm,
        )

        if result.p_value < alpha:
            false_positives += 1

    observed_fpr = false_positives / n_experiments

    # Monte Carlo standard error around alpha
    monte_carlo_se = np.sqrt(
        alpha * (1 - alpha) / n_experiments
    )

    upper_bound = alpha + 4 * monte_carlo_se

    assert observed_fpr <= upper_bound, (
        f"Observed Type I error {observed_fpr:.3%} "
        f"exceeded allowed bound {upper_bound:.3%}"
    )