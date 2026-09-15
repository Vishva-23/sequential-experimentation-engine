from dataclasses import dataclass

from consumer.assignment import (
    AssignmentResult,
    assign_card,
)
from producer.events import TransactionEvent


@dataclass(frozen=True)
class TransactionDecision:
    transaction_id: str
    card_id: str
    experiment_id: str

    arm: str
    fraud_threshold: float
    fraud_score: float

    approved: bool
    decision: str


def evaluate_transaction(
    event: TransactionEvent,
) -> TransactionDecision:

    assignment: AssignmentResult = assign_card(
        experiment_id=event.experiment_id,
        card_id=event.card_id,
    )

    approved = (
        event.fraud_score
        < assignment.fraud_threshold
    )

    decision = (
        "APPROVE"
        if approved
        else "DECLINE"
    )

    return TransactionDecision(
        transaction_id=event.transaction_id,
        card_id=event.card_id,
        experiment_id=event.experiment_id,
        arm=assignment.arm,
        fraud_threshold=(
            assignment.fraud_threshold
        ),
        fraud_score=event.fraud_score,
        approved=approved,
        decision=decision,
    )


if __name__ == "__main__":

    examples = [
        TransactionEvent(
            transaction_id="tx_001",
            card_id="card_0042",
            experiment_id="fraud_threshold_v1",
            timestamp="2026-09-15T16:00:00+00:00",
            amount=84.50,
            fraud_score=0.38,
        ),
        TransactionEvent(
            transaction_id="tx_002",
            card_id="card_0101",
            experiment_id="fraud_threshold_v1",
            timestamp="2026-09-15T16:00:01+00:00",
            amount=42.00,
            fraud_score=0.38,
        ),
    ]

    print("EXPERIMENT POLICY DECISIONS")
    print("-" * 65)

    for event in examples:

        result = evaluate_transaction(event)

        print(
            f"{result.transaction_id} | "
            f"{result.card_id} | "
            f"{result.arm:9} | "
            f"score={result.fraud_score:.2f} | "
            f"threshold={result.fraud_threshold:.2f} | "
            f"{result.decision}"
        )