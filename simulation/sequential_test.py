from dataclasses import dataclass

import numpy as np
from scipy.special import betaln


@dataclass
class SequentialTestResult:
    control_n: int
    treatment_n: int
    control_frauds: int
    treatment_frauds: int
    positive_discordants: int
    negative_discordants: int
    evidence: float
    threshold: float
    decision: str


class BernoulliMixtureEProcess:
    """
    Anytime-valid sequential test for two Bernoulli streams.

    Each update receives one control outcome and one treatment outcome.

    Outcomes:
        1 = fraud
        0 = non-fraud

    Under H0:
        p_control = p_treatment

    For discordant pairs, H0 implies that either direction is equally
    likely. We build a mixture likelihood ratio for those directions.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        prior_a: float = 100.0,
        prior_b: float = 100.0,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")

        if prior_a <= 0 or prior_b <= 0:
            raise ValueError("Prior parameters must be positive.")

        self.alpha = alpha
        self.threshold = 1.0 / alpha

        self.prior_a = prior_a
        self.prior_b = prior_b

        self.control_n = 0
        self.treatment_n = 0

        self.control_frauds = 0
        self.treatment_frauds = 0

        # Treatment fraud, control non-fraud
        self.positive_discordants = 0

        # Control fraud, treatment non-fraud
        self.negative_discordants = 0

    def _log_evidence(self) -> float:
        positive = self.positive_discordants
        negative = self.negative_discordants

        discordant_total = positive + negative

        if discordant_total == 0:
            return 0.0

        log_mixture_likelihood = (
            betaln(
                self.prior_a + positive,
                self.prior_b + negative,
            )
            - betaln(
                self.prior_a,
                self.prior_b,
            )
        )

        log_null_likelihood = (
            discordant_total * np.log(0.5)
        )

        return (
            log_mixture_likelihood
            - log_null_likelihood
        )

    @property
    def evidence(self) -> float:
        return float(np.exp(self._log_evidence()))

    @property
    def decision(self) -> str:
        if self.evidence >= self.threshold:
            return "STOP_REJECT_H0"

        return "CONTINUE"

    def update(
        self,
        control_outcome: int,
        treatment_outcome: int,
    ) -> SequentialTestResult:

        if control_outcome not in (0, 1):
            raise ValueError(
                "control_outcome must be 0 or 1."
            )

        if treatment_outcome not in (0, 1):
            raise ValueError(
                "treatment_outcome must be 0 or 1."
            )

        self.control_n += 1
        self.treatment_n += 1

        self.control_frauds += control_outcome
        self.treatment_frauds += treatment_outcome

        # Treatment fraud, control clean
        if control_outcome == 0 and treatment_outcome == 1:
            self.positive_discordants += 1

        # Control fraud, treatment clean
        elif control_outcome == 1 and treatment_outcome == 0:
            self.negative_discordants += 1

        return SequentialTestResult(
            control_n=self.control_n,
            treatment_n=self.treatment_n,
            control_frauds=self.control_frauds,
            treatment_frauds=self.treatment_frauds,
            positive_discordants=self.positive_discordants,
            negative_discordants=self.negative_discordants,
            evidence=self.evidence,
            threshold=self.threshold,
            decision=self.decision,
        )


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    control_rate = 0.035
    treatment_rate = 0.050

    test = BernoulliMixtureEProcess(
        alpha=0.05
    )

    max_observations = 50_000

    for observation in range(
        1,
        max_observations + 1,
    ):

        control_outcome = rng.binomial(
            n=1,
            p=control_rate,
        )

        treatment_outcome = rng.binomial(
            n=1,
            p=treatment_rate,
        )

        result = test.update(
            control_outcome=control_outcome,
            treatment_outcome=treatment_outcome,
        )

        if observation % 1_000 == 0:
            print(
                f"n={observation:,} | "
                f"control frauds={result.control_frauds} | "
                f"treatment frauds={result.treatment_frauds} | "
                f"evidence={result.evidence:.3f} | "
                f"decision={result.decision}"
            )

        if result.decision == "STOP_REJECT_H0":
            print("\nSTOPPING BOUNDARY CROSSED")
            print("-----------------------------")
            print(
                f"Observations per arm: "
                f"{observation:,}"
            )
            print(
                f"Evidence:             "
                f"{result.evidence:.3f}"
            )
            print(
                f"Boundary:             "
                f"{result.threshold:.3f}"
            )
            print(
                f"Control fraud rate:   "
                f"{result.control_frauds / result.control_n:.3%}"
            )
            print(
                f"Treatment fraud rate: "
                f"{result.treatment_frauds / result.treatment_n:.3%}"
            )

            break

    else:
        print("\nNo stopping boundary crossed.")