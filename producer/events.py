from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json


@dataclass(frozen=True)
class TransactionEvent:
    transaction_id: str
    card_id: str
    experiment_id: str
    timestamp: str
    amount: float
    fraud_score: float

    def __post_init__(self):
        if not self.transaction_id:
            raise ValueError(
                "transaction_id cannot be empty."
            )

        if not self.card_id:
            raise ValueError(
                "card_id cannot be empty."
            )

        if not self.experiment_id:
            raise ValueError(
                "experiment_id cannot be empty."
            )

        if self.amount < 0:
            raise ValueError(
                "amount cannot be negative."
            )

        if not 0 <= self.fraud_score <= 1:
            raise ValueError(
                "fraud_score must be between 0 and 1."
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass(frozen=True)
class FraudOutcomeEvent:
    transaction_id: str
    timestamp: str
    is_fraud: int

    def __post_init__(self):
        if not self.transaction_id:
            raise ValueError(
                "transaction_id cannot be empty."
            )

        if self.is_fraud not in (0, 1):
            raise ValueError(
                "is_fraud must be 0 or 1."
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def utc_timestamp() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()

if __name__ == "__main__":

    transaction = TransactionEvent(
        transaction_id="tx_000001",
        card_id="card_0042",
        experiment_id="fraud_threshold_v1",
        timestamp=utc_timestamp(),
        amount=84.50,
        fraud_score=0.38,
    )

    outcome = FraudOutcomeEvent(
        transaction_id="tx_000001",
        timestamp=utc_timestamp(),
        is_fraud=0,
    )

    print("TRANSACTION EVENT")
    print(transaction.to_json())

    print()

    print("FRAUD OUTCOME EVENT")
    print(outcome.to_json())