from dataclasses import dataclass

from consumer.outcome_join import ResolvedOutcome


@dataclass(frozen=True)
class ExperimentSnapshot:
    control_n: int
    control_fraud_leakage: int

    treatment_n: int
    treatment_fraud_leakage: int

    control_rate: float
    treatment_rate: float
    difference: float

    total_resolved: int


class OnlineExperimentStats:
    """
    Maintains sufficient statistics for the
    primary Bernoulli outcome.

    Y = 1 if an approved transaction is later
    confirmed as fraud, otherwise Y = 0.
    """

    def __init__(self):

        self.control_n = 0
        self.control_fraud_leakage = 0

        self.treatment_n = 0
        self.treatment_fraud_leakage = 0

    def update(
        self,
        outcome: ResolvedOutcome,
    ) -> None:

        if outcome.fraud_leakage not in (0, 1):
            raise ValueError(
                "fraud_leakage must be 0 or 1."
            )

        if outcome.arm == "CONTROL":

            self.control_n += 1

            self.control_fraud_leakage += (
                outcome.fraud_leakage
            )

        elif outcome.arm == "TREATMENT":

            self.treatment_n += 1

            self.treatment_fraud_leakage += (
                outcome.fraud_leakage
            )

        else:
            raise ValueError(
                f"Unknown experiment arm: "
                f"{outcome.arm}"
            )

    def snapshot(self) -> ExperimentSnapshot:

        if self.control_n == 0:
            control_rate = 0.0
        else:
            control_rate = (
                self.control_fraud_leakage
                / self.control_n
            )

        if self.treatment_n == 0:
            treatment_rate = 0.0
        else:
            treatment_rate = (
                self.treatment_fraud_leakage
                / self.treatment_n
            )

        difference = (
            treatment_rate - control_rate
        )

        total_resolved = (
            self.control_n
            + self.treatment_n
        )

        return ExperimentSnapshot(
            control_n=self.control_n,
            control_fraud_leakage=(
                self.control_fraud_leakage
            ),
            treatment_n=self.treatment_n,
            treatment_fraud_leakage=(
                self.treatment_fraud_leakage
            ),
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            difference=difference,
            total_resolved=total_resolved,
        )


if __name__ == "__main__":

    stats = OnlineExperimentStats()

    examples = [
        ResolvedOutcome(
            transaction_id="tx_001",
            card_id="card_001",
            experiment_id="fraud_threshold_v1",
            arm="CONTROL",
            approved=True,
            is_fraud=0,
            fraud_leakage=0,
        ),
        ResolvedOutcome(
            transaction_id="tx_002",
            card_id="card_002",
            experiment_id="fraud_threshold_v1",
            arm="CONTROL",
            approved=True,
            is_fraud=1,
            fraud_leakage=1,
        ),
        ResolvedOutcome(
            transaction_id="tx_003",
            card_id="card_003",
            experiment_id="fraud_threshold_v1",
            arm="TREATMENT",
            approved=True,
            is_fraud=1,
            fraud_leakage=1,
        ),
        ResolvedOutcome(
            transaction_id="tx_004",
            card_id="card_004",
            experiment_id="fraud_threshold_v1",
            arm="TREATMENT",
            approved=True,
            is_fraud=1,
            fraud_leakage=1,
        ),
    ]

    for outcome in examples:
        stats.update(outcome)

    snapshot = stats.snapshot()

    print("ONLINE EXPERIMENT STATE")
    print("-" * 50)

    print(
        f"Control:   "
        f"n={snapshot.control_n}, "
        f"fraud={snapshot.control_fraud_leakage}, "
        f"rate={snapshot.control_rate:.2%}"
    )

    print(
        f"Treatment: "
        f"n={snapshot.treatment_n}, "
        f"fraud={snapshot.treatment_fraud_leakage}, "
        f"rate={snapshot.treatment_rate:.2%}"
    )

    print(
        f"Difference: "
        f"{snapshot.difference * 100:.2f} pp"
    )

    print(
        f"Total resolved: "
        f"{snapshot.total_resolved}"
    )