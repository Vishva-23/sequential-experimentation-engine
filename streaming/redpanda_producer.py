import argparse
import heapq
import time
from datetime import datetime

from confluent_kafka import Producer

from producer.outcome_generator import OutcomeGenerator
from producer.transaction_generator import TransactionGenerator


TRANSACTION_TOPIC = "transaction.events"
OUTCOME_TOPIC = "fraud.outcomes"


class RedpandaEventPublisher:

    def __init__(
        self,
        bootstrap_servers: str = "localhost:19092",
    ):
        self.delivery_errors: list[str] = []

        self.producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "client.id": "sequential-experiment-producer",
            }
        )

    def _delivery_report(self, error, message) -> None:

        if error is not None:
            self.delivery_errors.append(str(error))

    def publish_transaction(self, event) -> None:

        self.producer.produce(
            topic=TRANSACTION_TOPIC,
            key=event.transaction_id,
            value=event.to_json(),
            on_delivery=self._delivery_report,
        )

        self.producer.poll(0)

    def publish_outcome(self, event) -> None:

        self.producer.produce(
            topic=OUTCOME_TOPIC,
            key=event.transaction_id,
            value=event.to_json(),
            on_delivery=self._delivery_report,
        )

        self.producer.poll(0)

    def flush(self, timeout: float = 10.0) -> None:

        remaining = self.producer.flush(timeout)

        if remaining != 0:
            raise RuntimeError(
                f"{remaining} Kafka messages were not delivered."
            )

        if self.delivery_errors:
            raise RuntimeError(
                "Kafka delivery failed: "
                + "; ".join(self.delivery_errors)
            )


def run_stream(
    count: int,
    transactions_per_second: float,
    delay_scale: float,
    bootstrap_servers: str,
) -> None:

    if count <= 0:
        raise ValueError("count must be positive.")

    if transactions_per_second <= 0:
        raise ValueError(
            "transactions_per_second must be positive."
        )

    if delay_scale < 0:
        raise ValueError(
            "delay_scale must be non-negative."
        )

    publisher = RedpandaEventPublisher(
        bootstrap_servers=bootstrap_servers
    )

    transaction_generator = TransactionGenerator(
        experiment_id="fraud_threshold_v1",
        seed=42,
    )

    outcome_generator = OutcomeGenerator(
        min_delay_seconds=60,
        max_delay_seconds=3600,
        seed=123,
    )

    pending_outcomes = []

    generated_count = 0
    published_outcomes = 0
    sequence_number = 0

    transaction_interval = (
        1.0 / transactions_per_second
    )

    next_transaction_time = time.monotonic()

    print("LIVE REDPANDA PRODUCER")
    print("-" * 60)

    print(f"Broker: {bootstrap_servers}")
    print(f"Transactions: {count}")
    print(
        "Transaction rate: "
        f"{transactions_per_second:.2f}/second"
    )
    print(f"Delay scale: {delay_scale}")

    print("-" * 60)

    while generated_count < count or pending_outcomes:

        now = time.monotonic()

        # Publish delayed outcomes that are now due.
        while (
            pending_outcomes
            and pending_outcomes[0][0] <= now
        ):

            _, _, outcome = heapq.heappop(
                pending_outcomes
            )

            publisher.publish_outcome(outcome)

            published_outcomes += 1

            print(
                "[OUTCOME] "
                f"{outcome.transaction_id} "
                f"fraud={outcome.is_fraud}"
            )

        # Generate the next transaction.
        if (
            generated_count < count
            and now >= next_transaction_time
        ):

            generated = transaction_generator.generate()

            transaction = generated.event

            publisher.publish_transaction(
                transaction
            )

            generated_count += 1

            print(
                "[TRANSACTION] "
                f"{transaction.transaction_id} "
                f"card={transaction.card_id} "
                f"amount={transaction.amount:.2f} "
                f"score={transaction.fraud_score:.3f}"
            )

            outcome = outcome_generator.generate(
                generated
            )

            transaction_timestamp = (
                datetime.fromisoformat(
                    transaction.timestamp
                )
            )

            outcome_timestamp = (
                datetime.fromisoformat(
                    outcome.timestamp
                )
            )

            simulated_delay_seconds = (
                outcome_timestamp
                - transaction_timestamp
            ).total_seconds()

            real_delay_seconds = (
                simulated_delay_seconds
                * delay_scale
            )

            sequence_number += 1

            heapq.heappush(
                pending_outcomes,
                (
                    time.monotonic()
                    + real_delay_seconds,
                    sequence_number,
                    outcome,
                ),
            )

            next_transaction_time += (
                transaction_interval
            )

        publisher.producer.poll(0)

        time.sleep(0.01)

    publisher.flush()

    print("-" * 60)
    print(
        f"Published transactions: {generated_count}"
    )
    print(
        f"Published outcomes: {published_outcomes}"
    )
    print(
        "All events delivered successfully."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--delay-scale",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:19092",
    )

    args = parser.parse_args()

    run_stream(
        count=args.count,
        transactions_per_second=args.rate,
        delay_scale=args.delay_scale,
        bootstrap_servers=args.bootstrap_servers,
    )