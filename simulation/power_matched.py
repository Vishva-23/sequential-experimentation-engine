import math

import numpy as np

from simulation.fixed_horizon import two_proportion_z_test
from simulation.power import required_sample_size


def simulate_stopping_time(
    control_rate: float,
    treatment_rate: float,
    max_n_per_arm: int,
    alpha: float = 0.05,
    prior_a: float = 100.0,
    prior_b: float = 100.0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    log_threshold = math.log(1.0 / alpha)

    discordant_probability = (
        control_rate * (1 - treatment_rate)
        + (1 - control_rate) * treatment_rate
    )

    positive_probability = (
        (1 - control_rate) * treatment_rate
        / discordant_probability
    )

    positive = 0
    negative = 0
    log_evidence = 0.0
    observation = 0

    while observation < max_n_per_arm:

        waiting_time = rng.geometric(
            discordant_probability
        )

        observation += waiting_time

        if observation > max_n_per_arm:
            break

        total_discordant = positive + negative

        direction = rng.binomial(
            n=1,
            p=positive_probability,
        )

        if direction == 1:
            predictive_probability = (
                prior_a + positive
            ) / (
                prior_a
                + prior_b
                + total_discordant
            )
            positive += 1

        else:
            predictive_probability = (
                prior_b + negative
            ) / (
                prior_a
                + prior_b
                + total_discordant
            )
            negative += 1

        log_evidence += math.log(
            predictive_probability / 0.5
        )

        if log_evidence >= log_threshold:

            if positive > negative:
                stop_direction = "TREATMENT_HIGHER"
            elif negative > positive:
                stop_direction = "CONTROL_HIGHER"
            else:
                stop_direction = "TIE"

            return {
                "stopped": True,
                "n_stop": observation,
                "direction": stop_direction,
            }

    return {
        "stopped": False,
        "n_stop": None,
        "direction": "NO_STOP",
    }


def calibrate_sequential_horizon(
    control_rate: float,
    treatment_rate: float,
    fixed_n: int,
    n_experiments: int = 5_000,
    target_power: float = 0.80,
    alpha: float = 0.05,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    simulation_cap = 20 * fixed_n

    true_direction = (
        "TREATMENT_HIGHER"
        if treatment_rate > control_rate
        else "CONTROL_HIGHER"
    )

    correct_stopping_times = []

    for _ in range(n_experiments):

        result = simulate_stopping_time(
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            max_n_per_arm=simulation_cap,
            alpha=alpha,
            rng=rng,
        )

        if (
            result["stopped"]
            and result["direction"] == true_direction
        ):
            correct_stopping_times.append(
                result["n_stop"]
            )

    required_correct = math.ceil(
        target_power * n_experiments
    )

    if len(correct_stopping_times) < required_correct:
        raise RuntimeError(
            "Simulation cap was too small to calibrate "
            "the requested power."
        )

    correct_stopping_times = np.sort(
        np.array(correct_stopping_times)
    )

    horizon = int(
        correct_stopping_times[
            required_correct - 1
        ]
    )

    return horizon


def evaluate_sequential(
    control_rate: float,
    treatment_rate: float,
    horizon: int,
    n_experiments: int = 5_000,
    alpha: float = 0.05,
    seed: int = 2026,
):
    rng = np.random.default_rng(seed)

    true_direction = (
        "TREATMENT_HIGHER"
        if treatment_rate > control_rate
        else "CONTROL_HIGHER"
    )

    correct = 0
    wrong = 0
    consumed_samples = []

    for _ in range(n_experiments):

        result = simulate_stopping_time(
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            max_n_per_arm=horizon,
            alpha=alpha,
            rng=rng,
        )

        if result["stopped"]:

            consumed_samples.append(
                result["n_stop"]
            )

            if result["direction"] == true_direction:
                correct += 1
            else:
                wrong += 1

        else:
            consumed_samples.append(
                horizon
            )

    consumed_samples = np.array(
        consumed_samples,
        dtype=float,
    )

    correct_rate = correct / n_experiments
    wrong_rate = wrong / n_experiments
    no_decision_rate = (
        1 - correct_rate - wrong_rate
    )

    # Monte Carlo uncertainty for estimated power
    power_se = math.sqrt(
        correct_rate
        * (1 - correct_rate)
        / n_experiments
    )

    power_ci_low = max(
        0.0,
        correct_rate - 1.96 * power_se,
    )

    power_ci_high = min(
        1.0,
        correct_rate + 1.96 * power_se,
    )

    mean_n = consumed_samples.mean()

    mean_n_se = (
        consumed_samples.std(ddof=1)
        / math.sqrt(n_experiments)
    )

    mean_n_ci_low = (
        mean_n - 1.96 * mean_n_se
    )

    mean_n_ci_high = (
        mean_n + 1.96 * mean_n_se
    )

    return {
        "correct_rate": correct_rate,
        "wrong_rate": wrong_rate,
        "no_decision_rate": no_decision_rate,
        "power_ci_low": power_ci_low,
        "power_ci_high": power_ci_high,
        "expected_n": mean_n,
        "median_n": np.median(consumed_samples),
        "mean_n_ci_low": mean_n_ci_low,
        "mean_n_ci_high": mean_n_ci_high,
    }


def evaluate_fixed_horizon(
    control_rate: float,
    treatment_rate: float,
    n_per_arm: int,
    n_experiments: int = 5_000,
    alpha: float = 0.05,
    seed: int = 2027,
):
    rng = np.random.default_rng(seed)

    true_direction = (
        "TREATMENT_HIGHER"
        if treatment_rate > control_rate
        else "CONTROL_HIGHER"
    )

    correct = 0
    wrong = 0

    for _ in range(n_experiments):

        control_frauds = rng.binomial(
            n=n_per_arm,
            p=control_rate,
        )

        treatment_frauds = rng.binomial(
            n=n_per_arm,
            p=treatment_rate,
        )

        result = two_proportion_z_test(
            control_successes=control_frauds,
            control_n=n_per_arm,
            treatment_successes=treatment_frauds,
            treatment_n=n_per_arm,
        )

        if result.p_value < alpha:

            observed_direction = (
                "TREATMENT_HIGHER"
                if result.absolute_difference > 0
                else "CONTROL_HIGHER"
            )

            if observed_direction == true_direction:
                correct += 1
            else:
                wrong += 1

    correct_rate = correct / n_experiments
    wrong_rate = wrong / n_experiments

    power_se = math.sqrt(
        correct_rate
        * (1 - correct_rate)
        / n_experiments
    )

    return {
        "correct_rate": correct_rate,
        "wrong_rate": wrong_rate,
        "power_ci_low": max(
            0.0,
            correct_rate - 1.96 * power_se,
        ),
        "power_ci_high": min(
            1.0,
            correct_rate + 1.96 * power_se,
        ),
    }


def run_comparison(
    control_rate: float,
    treatment_rate: float,
    n_calibration: int = 5_000,
    n_evaluation: int = 5_000,
    target_power: float = 0.80,
):
    fixed_n = required_sample_size(
        baseline_rate=control_rate,
        treatment_rate=treatment_rate,
        power=target_power,
    )

    # Stage 1: calibration
    horizon = calibrate_sequential_horizon(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        fixed_n=fixed_n,
        n_experiments=n_calibration,
        target_power=target_power,
        seed=42,
    )

    # Stage 2: completely independent evaluation
    sequential = evaluate_sequential(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        horizon=horizon,
        n_experiments=n_evaluation,
        seed=2026,
    )

    fixed = evaluate_fixed_horizon(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        n_per_arm=fixed_n,
        n_experiments=n_evaluation,
        seed=2027,
    )

    sample_saving = (
        1 - sequential["expected_n"] / fixed_n
    )

    return {
        "fixed_n": fixed_n,
        "horizon": horizon,
        "fixed": fixed,
        "sequential": sequential,
        "sample_saving": sample_saving,
    }


if __name__ == "__main__":

    control_rate = 0.035

    treatment_rates = [
        0.040,
        0.045,
        0.050,
    ]

    print("INDEPENDENT POWER-MATCHED VALIDATION")
    print("=" * 76)
    print("Calibration experiments: 5,000")
    print("Evaluation experiments:  5,000")

    for treatment_rate in treatment_rates:

        result = run_comparison(
            control_rate=control_rate,
            treatment_rate=treatment_rate,
        )

        seq = result["sequential"]
        fixed = result["fixed"]

        print()
        print(
            f"True rates:                 "
            f"{control_rate:.2%} vs "
            f"{treatment_rate:.2%}"
        )

        print(
            f"Fixed N per arm:            "
            f"{result['fixed_n']:,}"
        )

        print(
            f"Fixed empirical power:      "
            f"{fixed['correct_rate']:.2%} "
            f"[{fixed['power_ci_low']:.2%}, "
            f"{fixed['power_ci_high']:.2%}]"
        )

        print(
            f"Sequential frozen horizon:  "
            f"{result['horizon']:,}"
        )

        print(
            f"Sequential correct rate:    "
            f"{seq['correct_rate']:.2%} "
            f"[{seq['power_ci_low']:.2%}, "
            f"{seq['power_ci_high']:.2%}]"
        )

        print(
            f"Sequential wrong direction: "
            f"{seq['wrong_rate']:.2%}"
        )

        print(
            f"Sequential no decision:     "
            f"{seq['no_decision_rate']:.2%}"
        )

        print(
            f"Expected sequential N:      "
            f"{seq['expected_n']:,.0f} "
            f"[{seq['mean_n_ci_low']:,.0f}, "
            f"{seq['mean_n_ci_high']:,.0f}]"
        )

        print(
            f"Median sequential N:        "
            f"{seq['median_n']:,.0f}"
        )

        print(
            f"Sample saving vs fixed:     "
            f"{result['sample_saving']:.2%}"
        )