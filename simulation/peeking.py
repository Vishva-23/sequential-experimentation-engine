import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def simulate_peeking(
    n_experiments: int = 1_000,
    true_rate: float = 0.035,
    max_n_per_arm: int = 22_639,
    look_every: int = 100,
    alpha: float = 0.05,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # Sample sizes at which we "peek"
    look_sizes = np.arange(
        look_every,
        max_n_per_arm + 1,
        look_every,
    )

    # Make sure the exact fixed horizon is included
    if look_sizes[-1] != max_n_per_arm:
        look_sizes = np.append(
            look_sizes,
            max_n_per_arm,
        )

    # Number of NEW observations arriving between looks
    increments = np.diff(
        np.concatenate(([0], look_sizes))
    )

    n_looks = len(look_sizes)

    # Generate fraud counts for each new batch.
    # A/A means both arms have exactly the same true fraud rate.
    control_increments = rng.binomial(
        n=increments,
        p=true_rate,
        size=(n_experiments, n_looks),
    )

    treatment_increments = rng.binomial(
        n=increments,
        p=true_rate,
        size=(n_experiments, n_looks),
    )

    # Convert incremental counts into cumulative counts
    control_frauds = np.cumsum(
        control_increments,
        axis=1,
    )

    treatment_frauds = np.cumsum(
        treatment_increments,
        axis=1,
    )

    n = look_sizes[np.newaxis, :]

    # Observed fraud rates
    p_control = control_frauds / n
    p_treatment = treatment_frauds / n

    # Pooled fraud rate under H0
    pooled_rate = (
        control_frauds + treatment_frauds
    ) / (2 * n)

    # Standard error
    standard_error = np.sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * (2 / n)
    )

    # Z-statistics
    z_statistics = np.divide(
        p_treatment - p_control,
        standard_error,
        out=np.zeros_like(p_control, dtype=float),
        where=standard_error > 0,
    )

    # Two-sided p-values
    p_values = 2 * norm.sf(
        np.abs(z_statistics)
    )

    # Was this look significant?
    rejected = p_values < alpha

    # Fixed-horizon:
    # Only look at the FINAL test.
    fixed_horizon_fpr = rejected[:, -1].mean()

    # Naive peeking:
    # Has this experiment EVER produced p < alpha?
    ever_rejected = np.maximum.accumulate(
        rejected,
        axis=1,
    )

    cumulative_peeking_fpr = ever_rejected.mean(
        axis=0
    )

    naive_peeking_fpr = cumulative_peeking_fpr[-1]

    return {
        "look_sizes": look_sizes,
        "fixed_horizon_fpr": fixed_horizon_fpr,
        "naive_peeking_fpr": naive_peeking_fpr,
        "cumulative_peeking_fpr": cumulative_peeking_fpr,
    }


if __name__ == "__main__":

    results = simulate_peeking()

    fixed_fpr = results["fixed_horizon_fpr"]
    peeking_fpr = results["naive_peeking_fpr"]

    print("A/A MONTE CARLO")
    print("-----------------------------")
    print(f"Experiments:             {1_000:,}")
    print(f"True fraud rate:         {0.035:.2%}")
    print(f"Alpha:                   {0.05:.2%}")
    print(f"Maximum observations:    {22_639:,} per arm")
    print(f"Fixed-horizon FPR:       {fixed_fpr:.2%}")
    print(f"Naive-peeking FPR:       {peeking_fpr:.2%}")

    if fixed_fpr > 0:
        print(
            f"Inflation factor:        "
            f"{peeking_fpr / fixed_fpr:.2f}x"
        )

    # Plot Type I error accumulation
    plt.figure(figsize=(10, 6))

    plt.plot(
        results["look_sizes"],
        results["cumulative_peeking_fpr"],
        label="Naive repeated peeking",
    )

    plt.axhline(
        y=0.05,
        linestyle="--",
        label="Target alpha = 0.05",
    )

    plt.xlabel("Observations per arm")
    plt.ylabel(
        "Probability of ever falsely rejecting H0"
    )
    plt.title(
        "Type I Error Inflation from Repeated Peeking"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()