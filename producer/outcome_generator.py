from datetime import datetime, timedelta

import numpy as np

from producer.events import FraudOutcomeEvent
from producer.transaction_generator import (
    GeneratedTransaction,
    TransactionGenerator,
)


class OutcomeGenerator:

    def __init__(
        self,
        min_delay_seconds: int = 60,
        max_delay_seconds: int = 3600,
        seed: int = 123,
    ):
        if min_delay_seconds < 0:
            raise ValueError(
                "min_delay_seconds cannot be negative."
            )

        if max_delay_seconds < min_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be greater than "
                "or equal to min_delay_seconds."
            )

        self.min_delay_seconds = (
            min_delay_seconds
        )

        self.max_delay_seconds = (
            max_delay_seconds
        )

        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        generated_transaction: GeneratedTransaction,
    ) -> FraudOutcomeEvent:

        delay_seconds = int(
            self.rng.integers(
                self.min_delay_seconds,
                self.max_delay_seconds + 1,
            )
        )

        transaction_time = datetime.fromisoformat(
            generated_transaction.event.timestamp
        )

        outcome_time = (
            transaction_time
            + timedelta(seconds=delay_seconds)
        )

        return FraudOutcomeEvent(
            transaction_id=(
                generated_transaction
                .event
                .transaction_id
            ),
            timestamp=outcome_time.isoformat(),
            is_fraud=(
                generated_transaction
                .latent_is_fraud
            ),
        )


if __name__ == "__main__":

    transaction_generator = TransactionGenerator(
        base_fraud_rate=0.05,
        seed=42,
    )

    outcome_generator = OutcomeGenerator(
        min_delay_seconds=60,
        max_delay_seconds=3600,
        seed=123,
    )

    print("TRANSACTION → DELAYED FRAUD OUTCOME")
    print("-" * 75)

    for _ in range(5):

        generated = (
            transaction_generator.generate()
        )

        outcome = outcome_generator.generate(
            generated
        )

        transaction_time = datetime.fromisoformat(
            generated.event.timestamp
        )

        outcome_time = datetime.fromisoformat(
            outcome.timestamp
        )

        delay = (
            outcome_time - transaction_time
        ).total_seconds()

        print(
            f"{generated.event.transaction_id}"
        )

        print(
            f"  transaction time: "
            f"{generated.event.timestamp}"
        )

        print(
            f"  outcome time:     "
            f"{outcome.timestamp}"
        )

        print(
            f"  delay:            "
            f"{delay:.0f} seconds"
        )

        print(
            f"  fraud outcome:    "
            f"{outcome.is_fraud}"
        )

        print()