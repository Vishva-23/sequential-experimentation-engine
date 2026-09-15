from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln


@dataclass
class ConfidenceSequenceResult:
    control_rate: float
    treatment_rate: float
    difference: float

    control_lower: float
    control_upper: float

    treatment_lower: float
    treatment_upper: float

    difference_lower: float
    difference_upper: float


def bernoulli_log_e_value(
    successes: int,
    n: int,
    null_probability: float,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> float:

    if n <= 0:
        return 0.0

    if not 0 < null_probability < 1:
        return np.inf

    failures = n - successes

    log_mixture = (
        betaln(
            prior_a + successes,
            prior_b + failures,
        )
        - betaln(prior_a, prior_b)
    )

    log_null = (
        successes * np.log(null_probability)
        + failures * np.log1p(-null_probability)
    )

    return log_mixture - log_null


def bernoulli_confidence_sequence(
    successes: int,
    n: int,
    alpha: float = 0.025,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
):
    if n <= 0:
        return 0.0, 1.0

    observed_rate = successes / n

    log_threshold = np.log(1.0 / alpha)

    def boundary_function(p):
        return (
            bernoulli_log_e_value(
                successes=successes,
                n=n,
                null_probability=p,
                prior_a=prior_a,
                prior_b=prior_b,
            )
            - log_threshold
        )

    epsilon = 1e-12

    # Lower endpoint
    if successes == 0:
        lower = 0.0
    else:
        lower = brentq(
            boundary_function,
            epsilon,
            observed_rate,
        )

    # Upper endpoint
    if successes == n:
        upper = 1.0
    else:
        upper = brentq(
            boundary_function,
            observed_rate,
            1 - epsilon,
        )

    return lower, upper


def difference_confidence_sequence(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    alpha: float = 0.05,
) -> ConfidenceSequenceResult:

    if control_n <= 0 or treatment_n <= 0:
        raise ValueError(
            "Both arms must contain observations."
        )

    # Split alpha across the two arm-wise
    # confidence sequences.
    arm_alpha = alpha / 2

    control_lower, control_upper = (
        bernoulli_confidence_sequence(
            successes=control_successes,
            n=control_n,
            alpha=arm_alpha,
        )
    )

    treatment_lower, treatment_upper = (
        bernoulli_confidence_sequence(
            successes=treatment_successes,
            n=treatment_n,
            alpha=arm_alpha,
        )
    )

    control_rate = (
        control_successes / control_n
    )

    treatment_rate = (
        treatment_successes / treatment_n
    )

    difference = (
        treatment_rate - control_rate
    )

    # If:
    #
    # pT in [LT, UT]
    # pC in [LC, UC]
    #
    # then:
    #
    # pT - pC in [LT - UC, UT - LC]

    difference_lower = (
        treatment_lower - control_upper
    )

    difference_upper = (
        treatment_upper - control_lower
    )

    return ConfidenceSequenceResult(
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        difference=difference,

        control_lower=control_lower,
        control_upper=control_upper,

        treatment_lower=treatment_lower,
        treatment_upper=treatment_upper,

        difference_lower=difference_lower,
        difference_upper=difference_upper,
    )


if __name__ == "__main__":

    result = difference_confidence_sequence(
        control_successes=790,
        control_n=22_639,
        treatment_successes=906,
        treatment_n=22_639,
        alpha=0.05,
    )

    print("ANYTIME-VALID CONFIDENCE SEQUENCE")
    print("----------------------------------------")

    print(
        f"Control rate:      "
        f"{result.control_rate:.3%}"
    )

    print(
        f"Treatment rate:    "
        f"{result.treatment_rate:.3%}"
    )

    print(
        f"Observed difference: "
        f"{result.difference * 100:.3f} pp"
    )

    print()

    print(
        f"Control CS:        "
        f"[{result.control_lower:.3%}, "
        f"{result.control_upper:.3%}]"
    )

    print(
        f"Treatment CS:      "
        f"[{result.treatment_lower:.3%}, "
        f"{result.treatment_upper:.3%}]"
    )

    print()

    print(
        f"Difference CS:     "
        f"[{result.difference_lower * 100:.3f}, "
        f"{result.difference_upper * 100:.3f}] pp"
    )

    if (
        result.difference_lower > 0
        or result.difference_upper < 0
    ):
        print(
            "Zero excluded:     YES"
        )
    else:
        print(
            "Zero excluded:     NO"
        )