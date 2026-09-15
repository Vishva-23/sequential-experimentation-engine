from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class ProportionTestResult:
    control_rate: float
    treatment_rate: float
    absolute_difference: float
    z_statistic: float
    p_value: float


def two_proportion_z_test(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
) -> ProportionTestResult:

    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("Sample sizes must be positive.")

    # Observed fraud rates
    p_control = control_successes / control_n
    p_treatment = treatment_successes / treatment_n

    # Under H0, both groups are assumed to have the same true rate.
    # So we estimate that common rate by pooling both groups.
    pooled_successes = control_successes + treatment_successes
    pooled_n = control_n + treatment_n
    pooled_rate = pooled_successes / pooled_n

    # Standard error of the difference between the two proportions
    standard_error = np.sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * ((1 / control_n) + (1 / treatment_n))
    )

    if standard_error == 0:
        z_statistic = 0.0
        p_value = 1.0
    else:
        # How many standard errors apart are the two observed rates?
        z_statistic = (
            p_treatment - p_control
        ) / standard_error

        # Two-sided p-value
        p_value = 2 * norm.sf(abs(z_statistic))

    return ProportionTestResult(
        control_rate=p_control,
        treatment_rate=p_treatment,
        absolute_difference=p_treatment - p_control,
        z_statistic=z_statistic,
        p_value=p_value,
    )


if __name__ == "__main__":

    result = two_proportion_z_test(
        control_successes=790,
        control_n=22_639,
        treatment_successes=906,
        treatment_n=22_639,
    )

    print(f"Control fraud rate:   {result.control_rate:.3%}")
    print(f"Treatment fraud rate: {result.treatment_rate:.3%}")
    print(f"Difference:           {result.absolute_difference:.3%}")
    print(f"Z-statistic:          {result.z_statistic:.3f}")
    print(f"P-value:              {result.p_value:.5f}")

    if result.p_value < 0.05:
        print("Decision: Reject H0")
    else:
        print("Decision: Do not reject H0")