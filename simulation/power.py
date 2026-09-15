from math import ceil

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def required_sample_size(
    baseline_rate: float,
    treatment_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> int:

    effect_size = proportion_effectsize(
        baseline_rate,
        treatment_rate,
    )

    analysis = NormalIndPower()

    n_per_arm = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )

    return ceil(n_per_arm)


if __name__ == "__main__":

    baseline_rate = 0.035
    mde = 0.005

    treatment_rate = baseline_rate + mde

    n_per_arm = required_sample_size(
        baseline_rate=baseline_rate,
        treatment_rate=treatment_rate,
    )

    print(f"Baseline rate: {baseline_rate:.2%}")
    print(f"Treatment rate: {treatment_rate:.2%}")
    print(f"MDE: {mde:.2%}")
    print(f"Required sample size per arm: {n_per_arm:,}")
    print(f"Total sample size: {2 * n_per_arm:,}")