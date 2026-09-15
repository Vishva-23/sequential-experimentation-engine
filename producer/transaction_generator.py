from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from producer.events import (
    TransactionEvent,
    utc_timestamp,
)


@dataclass(frozen=True)
class GeneratedTransaction:
    event: TransactionEvent

    # Simulator-only latent truth.
    # This is NOT included in transaction.events.
    latent_is_fraud: int


class TransactionGenerator:

    def __init__(
        self,
        experiment_id: str = "fraud_threshold_v1",
        base_fraud_rate: float = 0.05,
        seed: int = 42,
    ):
        if not 0 < base_fraud_rate < 1:
            raise ValueError(
                "base_fraud_rate must be between 0 and 1."
            )

        self.experiment_id = experiment_id
        self.base_fraud_rate = base_fraud_rate
        self.rng = np.random.default_rng(seed)

        self.card_counter = 0

    def _generate_fraud_score(
        self,
        is_fraud: int,
    ) -> float:
        """
        Generate a fraud-model score correlated
        with the latent fraud outcome.
        """

        if is_fraud:
            score = self.rng.beta(
                4.0,
                3.0,
            )
        else:
            score = self.rng.beta(
                1.2,
                8.0,
            )

        return float(score)

    def generate(self) -> GeneratedTransaction:

        self.card_counter += 1

        card_id = (
            f"card_{self.card_counter:08d}"
        )

        transaction_id = (
            f"tx_{uuid4().hex}"
        )

        latent_is_fraud = int(
            self.rng.random()
            < self.base_fraud_rate
        )

        fraud_score = (
            self._generate_fraud_score(
                latent_is_fraud
            )
        )

        amount = float(
            np.round(
                self.rng.lognormal(
                    mean=3.5,
                    sigma=0.8,
                ),
                2,
            )
        )

        event = TransactionEvent(
            transaction_id=transaction_id,
            card_id=card_id,
            experiment_id=self.experiment_id,
            timestamp=utc_timestamp(),
            amount=amount,
            fraud_score=fraud_score,
        )

        return GeneratedTransaction(
            event=event,
            latent_is_fraud=latent_is_fraud,
        )


if __name__ == "__main__":

    generator = TransactionGenerator(
        seed=42,
    )

    print("SYNTHETIC PAYMENT STREAM")
    print("-" * 70)

    for _ in range(10):

        generated = generator.generate()

        print(
            generated.event.to_json()
        )

        print(
            "  simulator-only latent fraud:",
            generated.latent_is_fraud,
        )