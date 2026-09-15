import numpy as np
from scipy.special import betaln


def validate_confidence_sequence_coverage(
    n_experiments: int = 1_000,
    true_rate: float = 0.035,
    max_n_per_arm: int = 22_639,
    overall_alpha: float = 0.05,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
    batch_size: int = 100,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # Bonferroni split across control and treatment.
    arm_alpha = overall_alpha / 2

    log_threshold = np.log(
        1.0 / arm_alpha
    )

    log_prior_beta = betaln(
        prior_a,
        prior_b,
    )

    log_p = np.log(true_rate)
    log_one_minus_p = np.log1p(
        -true_rate
    )

    control_failures = 0
    treatment_failures = 0
    joint_failures = 0

    experiments_processed = 0

    while experiments_processed < n_experiments:

        current_batch = min(
            batch_size,
            n_experiments - experiments_processed,
        )

        control_outcomes = rng.binomial(
            n=1,
            p=true_rate,
            size=(
                current_batch,
                max_n_per_arm,
            ),
        )

        treatment_outcomes = rng.binomial(
            n=1,
            p=true_rate,
            size=(
                current_batch,
                max_n_per_arm,
            ),
        )

        control_successes = np.cumsum(
            control_outcomes,
            axis=1,
        )

        treatment_successes = np.cumsum(
            treatment_outcomes,
            axis=1,
        )

        n = np.arange(
            1,
            max_n_per_arm + 1,
        )[np.newaxis, :]

        control_fail_count = (
            n - control_successes
        )

        treatment_fail_count = (
            n - treatment_successes
        )

        control_log_e = (
            betaln(
                prior_a + control_successes,
                prior_b + control_fail_count,
            )
            - log_prior_beta
            - (
                control_successes * log_p
                + control_fail_count
                * log_one_minus_p
            )
        )

        treatment_log_e = (
            betaln(
                prior_a + treatment_successes,
                prior_b + treatment_fail_count,
            )
            - log_prior_beta
            - (
                treatment_successes * log_p
                + treatment_fail_count
                * log_one_minus_p
            )
        )

        control_ever_excluded = (
            control_log_e >= log_threshold
        ).any(axis=1)

        treatment_ever_excluded = (
            treatment_log_e >= log_threshold
        ).any(axis=1)

        joint_ever_failed = (
            control_ever_excluded
            | treatment_ever_excluded
        )

        control_failures += (
            control_ever_excluded.sum()
        )

        treatment_failures += (
            treatment_ever_excluded.sum()
        )

        joint_failures += (
            joint_ever_failed.sum()
        )

        experiments_processed += current_batch

    control_failure_rate = (
        control_failures / n_experiments
    )

    treatment_failure_rate = (
        treatment_failures / n_experiments
    )

    joint_failure_rate = (
        joint_failures / n_experiments
    )

    simultaneous_coverage = (
        1 - joint_failure_rate
    )

    return {
        "n_experiments": n_experiments,
        "control_failure_rate": (
            control_failure_rate
        ),
        "treatment_failure_rate": (
            treatment_failure_rate
        ),
        "joint_failure_rate": (
            joint_failure_rate
        ),
        "simultaneous_coverage": (
            simultaneous_coverage
        ),
        "arm_alpha": arm_alpha,
        "arm_boundary": (
            1.0 / arm_alpha
        ),
    }


if __name__ == "__main__":

    results = (
        validate_confidence_sequence_coverage()
    )

    print(
        "TIME-UNIFORM CONFIDENCE-SEQUENCE VALIDATION"
    )

    print("-" * 56)

    print(
        f"Experiments:               "
        f"{results['n_experiments']:,}"
    )

    print(
        "True control rate:         3.50%"
    )

    print(
        "True treatment rate:       3.50%"
    )

    print(
        "Overall alpha:             5.00%"
    )

    print(
        f"Arm-wise alpha:            "
        f"{results['arm_alpha']:.2%}"
    )

    print(
        f"Arm evidence boundary:     "
        f"{results['arm_boundary']:.1f}"
    )

    print()

    print(
        f"Control ever excluded:     "
        f"{results['control_failure_rate']:.2%}"
    )

    print(
        f"Treatment ever excluded:   "
        f"{results['treatment_failure_rate']:.2%}"
    )

    print(
        f"Either arm ever excluded:  "
        f"{results['joint_failure_rate']:.2%}"
    )

    print(
        f"Simultaneous coverage:     "
        f"{results['simultaneous_coverage']:.2%}"
    )