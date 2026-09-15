from dataclasses import dataclass

import numpy as np
from scipy.special import betaln


@dataclass(frozen=True)
class SRMResult:
    control_count: int
    treatment_count: int
    total_count: int
    observed_treatment_fraction: float
    expected_treatment_fraction: float
    log_evidence: float
    evidence: float
    threshold: float
    status: str


def srm_log_evidence(
    control_count: int,
    treatment_count: int,
    expected_treatment_fraction: float = 0.5,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> float:
    """
    Anytime-valid mixture e-value for detecting
    sample-ratio mismatch.

    Under H0, each randomized unit is assigned
    to treatment with the expected probability.
    """

    if control_count < 0 or treatment_count < 0:
        raise ValueError(
            "Assignment counts cannot be negative."
        )

    if not 0 < expected_treatment_fraction < 1:
        raise ValueError(
            "Expected treatment fraction must "
            "be between 0 and 1."
        )

    total_count = (
        control_count + treatment_count
    )

    if total_count == 0:
        return 0.0

    log_mixture = (
        betaln(
            prior_a + treatment_count,
            prior_b + control_count,
        )
        - betaln(prior_a, prior_b)
    )

    log_null = (
        treatment_count
        * np.log(expected_treatment_fraction)
        + control_count
        * np.log1p(
            -expected_treatment_fraction
        )
    )

    return log_mixture - log_null


def detect_srm(
    control_count: int,
    treatment_count: int,
    expected_treatment_fraction: float = 0.5,
    alpha: float = 0.001,
) -> SRMResult:

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    total_count = (
        control_count + treatment_count
    )

    if total_count == 0:
        observed_treatment_fraction = 0.0
    else:
        observed_treatment_fraction = (
            treatment_count / total_count
        )

    log_evidence = srm_log_evidence(
        control_count=control_count,
        treatment_count=treatment_count,
        expected_treatment_fraction=(
            expected_treatment_fraction
        ),
    )

    log_threshold = np.log(1.0 / alpha)

    # Avoid numerical overflow when evidence
    # becomes extremely large.
    evidence = float(
        np.exp(
            min(log_evidence, 700)
        )
    )

    threshold = 1.0 / alpha

    if log_evidence >= log_threshold:
        status = "INVALID_EXPERIMENT"
    else:
        status = "VALID"

    return SRMResult(
        control_count=control_count,
        treatment_count=treatment_count,
        total_count=total_count,
        observed_treatment_fraction=(
            observed_treatment_fraction
        ),
        expected_treatment_fraction=(
            expected_treatment_fraction
        ),
        log_evidence=log_evidence,
        evidence=evidence,
        threshold=threshold,
        status=status,
    )


if __name__ == "__main__":

    healthy = detect_srm(
        control_count=5_020,
        treatment_count=4_980,
    )

    broken = detect_srm(
        control_count=7_000,
        treatment_count=3_000,
    )

    print("HEALTHY EXPERIMENT")
    print("-" * 45)
    print(
        f"Control:             "
        f"{healthy.control_count:,}"
    )
    print(
        f"Treatment:           "
        f"{healthy.treatment_count:,}"
    )
    print(
        f"Treatment fraction:  "
        f"{healthy.observed_treatment_fraction:.2%}"
    )
    print(
        f"SRM evidence:        "
        f"{healthy.evidence:.4f}"
    )
    print(
        f"SRM boundary:        "
        f"{healthy.threshold:.0f}"
    )
    print(
        f"Status:              "
        f"{healthy.status}"
    )

    print()
    print("BROKEN EXPERIMENT")
    print("-" * 45)
    print(
        f"Control:             "
        f"{broken.control_count:,}"
    )
    print(
        f"Treatment:           "
        f"{broken.treatment_count:,}"
    )
    print(
        f"Treatment fraction:  "
        f"{broken.observed_treatment_fraction:.2%}"
    )
    print(
        f"log(SRM evidence):   "
        f"{broken.log_evidence:.2f}"
    )
    print(
        f"SRM boundary:        "
        f"{broken.threshold:.0f}"
    )
    print(
        f"Status:              "
        f"{broken.status}"
    )