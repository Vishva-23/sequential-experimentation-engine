from dataclasses import dataclass

from consumer.pair_outcomes import CompletedPair
from simulation.sequential_test import (
    BernoulliMixtureEProcess,
    SequentialTestResult,
)


@dataclass(frozen=True)
class StreamingInferenceSnapshot:
    completed_pairs: int

    control_leakage: int
    treatment_leakage: int

    positive_discordants: int
    negative_discordants: int

    evidence: float
    threshold: float
    decision: str


class StreamingSequentialInference:
    """
    Adapter between completed streaming pairs and the
    validated Bernoulli mixture e-process.

    Each CompletedPair is passed exactly once from the
    upstream PairOutcomeBuffer.

    Primary outcome:

        Y = 1
            if the transaction was approved and later
            confirmed as fraudulent.

        Y = 0
            otherwise.

    Pair membership is fixed before fraud outcomes are
    observed.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        prior_a: float = 100.0,
        prior_b: float = 100.0,
    ):

        self.test = BernoulliMixtureEProcess(
            alpha=alpha,
            prior_a=prior_a,
            prior_b=prior_b,
        )

    def update(
        self,
        completed_pair: CompletedPair,
    ) -> StreamingInferenceSnapshot:

        if completed_pair.control_y not in (0, 1):
            raise ValueError(
                "control_y must be 0 or 1."
            )

        if completed_pair.treatment_y not in (0, 1):
            raise ValueError(
                "treatment_y must be 0 or 1."
            )

        result = self.test.update(
            control_outcome=completed_pair.control_y,
            treatment_outcome=completed_pair.treatment_y,
        )

        return self._snapshot_from_result(
            result
        )

    def snapshot(
        self,
    ) -> StreamingInferenceSnapshot:

        return StreamingInferenceSnapshot(
            completed_pairs=self.test.control_n,
            control_leakage=self.test.control_frauds,
            treatment_leakage=self.test.treatment_frauds,
            positive_discordants=(
                self.test.positive_discordants
            ),
            negative_discordants=(
                self.test.negative_discordants
            ),
            evidence=self.test.evidence,
            threshold=self.test.threshold,
            decision=self.test.decision,
        )

    @staticmethod
    def _snapshot_from_result(
        result: SequentialTestResult,
    ) -> StreamingInferenceSnapshot:

        return StreamingInferenceSnapshot(
            completed_pairs=result.control_n,
            control_leakage=result.control_frauds,
            treatment_leakage=result.treatment_frauds,
            positive_discordants=(
                result.positive_discordants
            ),
            negative_discordants=(
                result.negative_discordants
            ),
            evidence=result.evidence,
            threshold=result.threshold,
            decision=result.decision,
        )


if __name__ == "__main__":

    inference = StreamingSequentialInference(
        alpha=0.05,
        prior_a=100.0,
        prior_b=100.0,
    )

    examples = [
        CompletedPair(
            pair_id=1,
            control_transaction_id="tx_c1",
            treatment_transaction_id="tx_t1",
            control_y=0,
            treatment_y=1,
        ),
        CompletedPair(
            pair_id=2,
            control_transaction_id="tx_c2",
            treatment_transaction_id="tx_t2",
            control_y=0,
            treatment_y=0,
        ),
        CompletedPair(
            pair_id=3,
            control_transaction_id="tx_c3",
            treatment_transaction_id="tx_t3",
            control_y=1,
            treatment_y=0,
        ),
        CompletedPair(
            pair_id=4,
            control_transaction_id="tx_c4",
            treatment_transaction_id="tx_t4",
            control_y=1,
            treatment_y=1,
        ),
    ]

    print("STREAMING SEQUENTIAL INFERENCE")
    print("-" * 60)

    for pair in examples:

        result = inference.update(
            pair
        )

        print(
            f"Pair {pair.pair_id}: "
            f"({pair.control_y}, "
            f"{pair.treatment_y})"
        )

        print(
            f"  positive discordants = "
            f"{result.positive_discordants}"
        )

        print(
            f"  negative discordants = "
            f"{result.negative_discordants}"
        )

        print(
            f"  evidence = "
            f"{result.evidence:.6f}"
        )

        print(
            f"  decision = "
            f"{result.decision}"
        )

        print()

    final = inference.snapshot()

    print("FINAL STATE")
    print("-" * 60)

    print(
        "Completed pairs:",
        final.completed_pairs,
    )

    print(
        "Control leakage:",
        final.control_leakage,
    )

    print(
        "Treatment leakage:",
        final.treatment_leakage,
    )

    print(
        "Positive discordants:",
        final.positive_discordants,
    )

    print(
        "Negative discordants:",
        final.negative_discordants,
    )

    print(
        "Evidence:",
        f"{final.evidence:.6f}",
    )

    print(
        "Boundary:",
        f"{final.threshold:.1f}",
    )

    print(
        "Decision:",
        final.decision,
    )