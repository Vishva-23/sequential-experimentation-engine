import numpy as np
from scipy.special import betaln


def validate_sequential_type1_error(
    n_experiments: int = 1_000,
    true_rate: float = 0.035,
    max_n_per_arm: int = 22_639,
    alpha: float = 0.05,
    prior_a: float = 100.0,
    prior_b: float = 100.0,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    threshold = 1.0 / alpha

    # Under A/A:
    # P(control != treatment)
    discordant_probability = (
        2 * true_rate * (1 - true_rate)
    )

    # Number of discordant pairs occurring in each experiment
    n_discordant = rng.binomial(
        n=max_n_per_arm,
        p=discordant_probability,
        size=n_experiments,
    )

    max_discordant = n_discordant.max()

    # Under H0, conditional on a discordant pair,
    # either direction has probability 0.5.
    directions = rng.binomial(
        n=1,
        p=0.5,
        size=(n_experiments, max_discordant),
    )

    positive = np.cumsum(directions, axis=1)

    discordant_index = np.arange(
        1,
        max_discordant + 1,
    )

    negative = (
        discordant_index[np.newaxis, :]
        - positive
    )

    # Mixture log likelihood
    log_mixture = (
        betaln(
            prior_a + positive,
            prior_b + negative,
        )
        - betaln(prior_a, prior_b)
    )

    # Under H0, each discordant direction has probability 0.5
    log_null = (
        discordant_index[np.newaxis, :]
        * np.log(0.5)
    )

    log_evidence = log_mixture - log_null

    log_threshold = np.log(threshold)

    crossed = log_evidence >= log_threshold

    # Ignore discordant observations that occur beyond
    # the actual number observed in each experiment.
    valid_positions = (
        discordant_index[np.newaxis, :]
        <= n_discordant[:, np.newaxis]
    )

    crossed &= valid_positions

    # Did each experiment EVER cross the boundary?
    ever_crossed = crossed.any(axis=1)

    sequential_fpr = ever_crossed.mean()

    false_stops = ever_crossed.sum()

    return {
        "n_experiments": n_experiments,
        "false_stops": false_stops,
        "sequential_fpr": sequential_fpr,
        "threshold": threshold,
        "mean_discordant": n_discordant.mean(),
    }


if __name__ == "__main__":

    results = validate_sequential_type1_error()

    print("ANYTIME-VALID A/A MONTE CARLO")
    print("--------------------------------")
    print(
        f"Experiments:             "
        f"{results['n_experiments']:,}"
    )
    print("True control rate:       3.50%")
    print("True treatment rate:     3.50%")
    print("Alpha:                   5.00%")
    print(
        f"Evidence boundary:       "
        f"{results['threshold']:.1f}"
    )
    print(
        f"False stops:             "
        f"{results['false_stops']}"
    )
    print(
        f"Sequential FPR:          "
        f"{results['sequential_fpr']:.2%}"
    )
    print(
        f"Mean discordant pairs:   "
        f"{results['mean_discordant']:.1f}"
    )

    print()
    print("COMPARISON")
    print("--------------------------------")
    print("Nominal alpha:           5.00%")
    print("Naive peeking (earlier): 42.00%")
    print(
        f"Anytime-valid test:      "
        f"{results['sequential_fpr']:.2%}"
    )