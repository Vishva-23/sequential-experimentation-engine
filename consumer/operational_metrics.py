from dataclasses import dataclass

from consumer.policy import TransactionDecision


@dataclass(frozen=True)
class OperationalMetricsSnapshot:
    control_n: int
    control_approved: int

    treatment_n: int
    treatment_approved: int

    control_approval_rate: float
    treatment_approval_rate: float
    approval_rate_difference: float

    total_transactions: int


class OnlineOperationalMetrics:
    """
    Maintains immediate operational metrics using
    eligible randomized transaction decisions.

    These metrics do not require delayed fraud labels.

    Approval rate is treated as a business KPI rather
    than as the primary statistical decision rule.
    """

    def __init__(self):

        self.control_n = 0
        self.control_approved = 0

        self.treatment_n = 0
        self.treatment_approved = 0

    def update(
        self,
        decision: TransactionDecision,
    ) -> None:

        approved = int(
            decision.approved
        )

        if decision.arm == "CONTROL":

            self.control_n += 1
            self.control_approved += approved

        elif decision.arm == "TREATMENT":

            self.treatment_n += 1
            self.treatment_approved += approved

        else:

            raise ValueError(
                f"Unknown experiment arm: "
                f"{decision.arm}"
            )

    def snapshot(
        self,
    ) -> OperationalMetricsSnapshot:

        if self.control_n == 0:
            control_approval_rate = 0.0
        else:
            control_approval_rate = (
                self.control_approved
                / self.control_n
            )

        if self.treatment_n == 0:
            treatment_approval_rate = 0.0
        else:
            treatment_approval_rate = (
                self.treatment_approved
                / self.treatment_n
            )

        approval_rate_difference = (
            treatment_approval_rate
            - control_approval_rate
        )

        return OperationalMetricsSnapshot(
            control_n=self.control_n,
            control_approved=self.control_approved,
            treatment_n=self.treatment_n,
            treatment_approved=self.treatment_approved,
            control_approval_rate=(
                control_approval_rate
            ),
            treatment_approval_rate=(
                treatment_approval_rate
            ),
            approval_rate_difference=(
                approval_rate_difference
            ),
            total_transactions=(
                self.control_n
                + self.treatment_n
            ),
        )


if __name__ == "__main__":

    from producer.events import TransactionEvent
    from consumer.policy import evaluate_transaction

    metrics = OnlineOperationalMetrics()

    control_event = TransactionEvent(
        transaction_id="tx_control",
        card_id="card_0101",
        experiment_id="fraud_threshold_v1",
        timestamp="2026-09-15T20:00:00+00:00",
        amount=75.00,
        fraud_score=0.38,
    )

    treatment_event = TransactionEvent(
        transaction_id="tx_treatment",
        card_id="card_0042",
        experiment_id="fraud_threshold_v1",
        timestamp="2026-09-15T20:00:01+00:00",
        amount=75.00,
        fraud_score=0.38,
    )

    metrics.update(
        evaluate_transaction(control_event)
    )

    metrics.update(
        evaluate_transaction(treatment_event)
    )

    state = metrics.snapshot()

    print("OPERATIONAL METRICS")
    print("-" * 50)

    print(
        "Control approval rate:",
        f"{state.control_approval_rate:.2%}",
    )

    print(
        "Treatment approval rate:",
        f"{state.treatment_approval_rate:.2%}",
    )

    print(
        "Difference:",
        f"{state.approval_rate_difference * 100:.2f} pp",
    )