import math

import numpy as np

from simulation.power import required_sample_size


def simulate_sequential_experiment(
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

    threshold = 1.0 / alpha
    log_threshold = math.log(threshold)

    # Probability that a control/treatment pair is discordant
    discordant_probability = (
        control_rate * (1 - treatment_rate)
        + (1 - control_rate) * treatment_rate
    )

    # Given that the pair is discordant,
    # probability it is:
    #
    # control = 0, treatment = 1
    positive_probability = (
        (1 - control_rate) * treatment_rate
        / discordant_probability
    )

    positive = 0
    negative = 0
    log_evidence = 0.0

    observation = 0

    while observation < max_n_per_arm:

        # Number of transaction pairs until the next
        # informative discordant pair
        waiting_time = rng.geometric(
            discordant_probability
        )

        observation += waiting_time

        if observation > max_n_per_arm:
            break

        direction = rng.binomial(
            n=1,
            p=positive_probability,
        )

        total_discordant = positive + negative

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

        # Under H0, either discordant direction
        # has probability 0.5.
        evidence_multiplier = (
            predictive_probability / 0.5
        )

        log_evidence += math.log(
            evidence_multiplier
        )

        if log_evidence >= log_threshold:

            return {
                "stopped": True,
                "n_stop": observation,
                "evidence": math.exp(log_evidence),
            }

    return {
        "stopped": False,
        "n_stop": max_n_per_arm,
        "evidence": math.exp(log_evidence),
    }


def estimate_asn(
    control_rate: float,
    treatment_rate: float,
    n_experiments: int = 1_000,
    alpha: float = 0.05,
    power: float = 0.80,
    seed: int = 42,
):

    fixed_n = required_sample_size(
        baseline_rate=control_rate,
        treatment_rate=treatment_rate,
        alpha=alpha,
        power=power,
    )

    rng = np.random.default_rng(seed)

    sample_numbers = []
    stopped = []

    for _ in range(n_experiments):

        result = simulate_sequential_experiment(
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            max_n_per_arm=fixed_n,
            alpha=alpha,
            rng=rng,
        )

        sample_numbers.append(
            result["n_stop"]
        )

        stopped.append(
            result["stopped"]
        )

    sample_numbers = np.array(sample_numbers)
    stopped = np.array(stopped)

    restricted_asn = sample_numbers.mean()
    median_n = np.median(sample_numbers)

    stopping_rate = stopped.mean()

    apparent_saving = (
        1 - restricted_asn / fixed_n
    )

    return {
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "fixed_n": fixed_n,
        "restricted_asn": restricted_asn,
        "median_n": median_n,
        "stopping_rate": stopping_rate,
        "apparent_saving": apparent_saving,
    }


if __name__ == "__main__":

    control_rate = 0.035

    treatment_rates = [
        0.040,
        0.045,
        0.050,
    ]

    print("SEQUENTIAL EFFICIENCY MONTE CARLO")
    print("=" * 72)

    for treatment_rate in treatment_rates:

        result = estimate_asn(
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            n_experiments=1_000,
        )

        print()
        print(
            f"True rates: "
            f"{control_rate:.2%} vs "
            f"{treatment_rate:.2%}"
        )

        print(
            f"Fixed N per arm:        "
            f"{result['fixed_n']:,}"
        )

        print(
            f"Restricted mean N:      "
            f"{result['restricted_asn']:,.0f}"
        )

        print(
            f"Median N:               "
            f"{result['median_n']:,.0f}"
        )

        print(
            f"Stopped by fixed N:     "
            f"{result['stopping_rate']:.2%}"
        )

        print(
            f"Apparent sample saving: "
            f"{result['apparent_saving']:.2%}"
        )