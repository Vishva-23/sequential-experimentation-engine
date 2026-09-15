from dataclasses import dataclass

from consumer.policy import (
    TransactionDecision,
    evaluate_transaction,
)
from producer.events import (
    FraudOutcomeEvent,
    TransactionEvent,
)


@dataclass(frozen=True)
class ResolvedOutcome:
    transaction_id: str
    card_id: str
    experiment_id: str

    arm: str
    approved: bool
    is_fraud: int

    # Primary Bernoulli experimental outcome:
    # 1 = approved fraudulent transaction
    # 0 = otherwise
    fraud_leakage: int


class OutcomeJoinState:
    """
    Joins transaction decisions with delayed fraud
    outcomes using transaction_id.

    Supports either arrival order:

        transaction -> outcome
        outcome -> transaction

    Duplicate resolved events are ignored.
    """

    def __init__(self):
        self.pending_decisions: dict[
            str,
            TransactionDecision,
        ] = {}

        self.pending_outcomes: dict[
            str,
            FraudOutcomeEvent,
        ] = {}

        self.resolved_transaction_ids: set[str] = set()

    def process_transaction(
        self,
        event: TransactionEvent,
    ) -> ResolvedOutcome | None:

        transaction_id = event.transaction_id

        if (
            transaction_id
            in self.resolved_transaction_ids
        ):
            return None

        decision = evaluate_transaction(event)

        existing = self.pending_decisions.get(
            transaction_id
        )

        if existing is not None:
            if existing != decision:
                raise ValueError(
                    "Conflicting duplicate transaction "
                    f"for {transaction_id}."
                )

            return None

        self.pending_decisions[
            transaction_id
        ] = decision

        outcome = self.pending_outcomes.pop(
            transaction_id,
            None,
        )

        if outcome is None:
            return None

        return self._resolve(
            transaction_id=transaction_id,
            outcome=outcome,
        )

    def process_outcome(
        self,
        outcome: FraudOutcomeEvent,
    ) -> ResolvedOutcome | None:

        transaction_id = outcome.transaction_id

        if (
            transaction_id
            in self.resolved_transaction_ids
        ):
            return None

        decision = self.pending_decisions.get(
            transaction_id
        )

        if decision is None:

            existing = self.pending_outcomes.get(
                transaction_id
            )

            if existing is not None:
                if existing != outcome:
                    raise ValueError(
                        "Conflicting duplicate outcome "
                        f"for {transaction_id}."
                    )

                return None

            self.pending_outcomes[
                transaction_id
            ] = outcome

            return None

        return self._resolve(
            transaction_id=transaction_id,
            outcome=outcome,
        )

    def _resolve(
        self,
        transaction_id: str,
        outcome: FraudOutcomeEvent,
    ) -> ResolvedOutcome:

        decision = self.pending_decisions.pop(
            transaction_id
        )

        fraud_leakage = int(
            decision.approved
            and outcome.is_fraud == 1
        )

        resolved = ResolvedOutcome(
            transaction_id=transaction_id,
            card_id=decision.card_id,
            experiment_id=decision.experiment_id,
            arm=decision.arm,
            approved=decision.approved,
            is_fraud=outcome.is_fraud,
            fraud_leakage=fraud_leakage,
        )

        self.resolved_transaction_ids.add(
            transaction_id
        )

        return resolved


if __name__ == "__main__":

    join_state = OutcomeJoinState()

    transaction = TransactionEvent(
        transaction_id="tx_demo",
        card_id="card_0042",
        experiment_id="fraud_threshold_v1",
        timestamp="2026-09-15T16:00:00+00:00",
        amount=84.50,
        fraud_score=0.38,
    )

    outcome = FraudOutcomeEvent(
        transaction_id="tx_demo",
        timestamp="2026-09-15T16:30:00+00:00",
        is_fraud=1,
    )

    print("1. TRANSACTION ARRIVES")
    print("-" * 50)

    result = join_state.process_transaction(
        transaction
    )

    print("Resolved:", result)
    print(
        "Pending decisions:",
        len(join_state.pending_decisions),
    )

    print()
    print("2. FRAUD OUTCOME ARRIVES")
    print("-" * 50)

    result = join_state.process_outcome(
        outcome
    )

    print("Resolved:", result)

    print(
        "Pending decisions:",
        len(join_state.pending_decisions),
    )

    print(
        "Primary outcome Y:",
        result.fraud_leakage,
    )