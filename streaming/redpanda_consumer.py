import argparse
import json

from confluent_kafka import Consumer

from consumer.experiment_engine import ExperimentEngine
from persistence.postgres import PostgresExperimentRepository
from producer.events import FraudOutcomeEvent, TransactionEvent


TRANSACTION_TOPIC = "transaction.events"
OUTCOME_TOPIC = "fraud.outcomes"

DEFAULT_DATABASE_URL = (
    "postgresql://experiment:experiment"
    "@localhost:5432/experiments"
)


class ExperimentStreamConsumer:
    """
    Durable Redpanda consumer for the sequential
    experimentation engine.

    Processing order for each new Kafka event:

        Kafka event
            -> ExperimentEngine
            -> PostgreSQL event + state transaction
            -> Kafka offset commit

    PostgreSQL commits before the Kafka offset.

    If the process crashes after PostgreSQL commits but
    before Kafka commits, Redpanda may redeliver the
    message. The processed_events table identifies the
    event by:

        (topic, partition, offset)

    and prevents it from being applied twice.

    On startup, persisted events are replayed in their
    original processing order to reconstruct the
    in-memory ExperimentEngine.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:19092",
        group_id: str = (
            "sequential-experiment-engine-postgres-v1"
        ),
        database_url: str = DEFAULT_DATABASE_URL,
    ):
        self.engine = ExperimentEngine()

        self.repository = PostgresExperimentRepository(
            database_url
        )

        self.repository.ensure_schema()

        # Rebuild in-memory experiment state from the
        # durable PostgreSQL event log before consuming
        # new Kafka messages.
        self._restore_engine()

        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )

        self.consumer.subscribe(
            [
                TRANSACTION_TOPIC,
                OUTCOME_TOPIC,
            ]
        )

    @staticmethod
    def _decode_json(message) -> dict:
        value = message.value()

        if value is None:
            raise ValueError(
                "Kafka message has no value."
            )

        return json.loads(
            value.decode("utf-8")
        )

    @staticmethod
    def _transaction_from_payload(
        payload: dict,
    ) -> TransactionEvent:
        return TransactionEvent(
            transaction_id=payload[
                "transaction_id"
            ],
            card_id=payload["card_id"],
            experiment_id=payload[
                "experiment_id"
            ],
            timestamp=payload["timestamp"],
            amount=float(
                payload["amount"]
            ),
            fraud_score=float(
                payload["fraud_score"]
            ),
        )

    @staticmethod
    def _outcome_from_payload(
        payload: dict,
    ) -> FraudOutcomeEvent:
        return FraudOutcomeEvent(
            transaction_id=payload[
                "transaction_id"
            ],
            timestamp=payload["timestamp"],
            is_fraud=int(
                payload["is_fraud"]
            ),
        )

    def _process_payload(
        self,
        topic: str,
        payload: dict,
    ):
        if topic == TRANSACTION_TOPIC:
            event = self._transaction_from_payload(
                payload
            )

            return self.engine.process_transaction(
                event
            )

        if topic == OUTCOME_TOPIC:
            event = self._outcome_from_payload(
                payload
            )

            return self.engine.process_outcome(
                event
            )

        raise ValueError(
            f"Unexpected topic: {topic}"
        )

    def _restore_engine(self) -> None:
        """
        Reconstruct the complete in-memory experiment
        state from the durable PostgreSQL event log.
        """

        stored_events = self.repository.load_events()

        print(
            "Restoring experiment state from "
            "PostgreSQL..."
        )

        if not stored_events:
            print(
                "No persisted events found. "
                "Starting from empty state."
            )
            return

        for stored_event in stored_events:
            self._process_payload(
                topic=stored_event.topic,
                payload=stored_event.payload,
            )

        state = self.engine.snapshot()

        print(
            f"Replayed {len(stored_events)} "
            "persisted events."
        )

        print(
            "Restored randomized:",
            state.total_randomized,
        )

        print(
            "Restored resolved:",
            state.resolved.total_resolved,
        )

        print(
            "Restored completed pairs:",
            state.inference.completed_pairs,
        )

        print(
            "Restored evidence:",
            f"{state.inference.evidence:.6f}",
        )

        print(
            "Restored SRM:",
            state.srm.status,
        )

        print(
            "Restored guardrails:",
            state.guardrails.decision,
        )

        print(
            "Restored decision:",
            state.decision,
        )

    @staticmethod
    def _message_key(
        message,
    ) -> str | None:
        key = message.key()

        if key is None:
            return None

        return key.decode("utf-8")

    @staticmethod
    def _print_state(
        topic,
        state,
    ) -> None:
        print()

        print(
            f"[{topic}]"
        )

        print(
            "Randomized:",
            state.total_randomized,
            "| Resolved:",
            state.resolved.total_resolved,
            "| Pairs:",
            state.inference.completed_pairs,
        )

        print(
            "Approval C/T:",
            f"{state.operational.control_approval_rate:.2%}",
            "/",
            f"{state.operational.treatment_approval_rate:.2%}",
        )

        print(
            "Leakage C/T:",
            f"{state.resolved.control_rate:.2%}",
            "/",
            f"{state.resolved.treatment_rate:.2%}",
        )

        print(
            "Evidence:",
            f"{state.inference.evidence:.4f}",
            "/",
            f"{state.inference.threshold:.1f}",
        )

        print(
            "SRM:",
            state.srm.status,
            "| Guardrails:",
            state.guardrails.decision,
            "| ENGINE:",
            state.decision,
        )

    def run(
        self,
        max_messages: int | None = None,
    ) -> None:
        processed = 0
        duplicates = 0

        print()
        print(
            "DURABLE REDPANDA EXPERIMENT CONSUMER"
        )
        print("-" * 60)

        print("Listening to:")
        print(
            f"  {TRANSACTION_TOPIC}"
        )
        print(
            f"  {OUTCOME_TOPIC}"
        )

        print(
            "Persistence: PostgreSQL"
        )

        print("-" * 60)

        try:
            while True:
                message = self.consumer.poll(
                    timeout=1.0
                )

                if message is None:
                    continue

                if message.error():
                    raise RuntimeError(
                        str(message.error())
                    )

                topic = message.topic()
                partition_id = message.partition()
                kafka_offset = message.offset()

                # -----------------------------------------
                # Durable event idempotency
                # -----------------------------------------

                if self.repository.event_exists(
                    topic=topic,
                    partition_id=partition_id,
                    kafka_offset=kafka_offset,
                ):
                    duplicates += 1

                    print(
                        "[DUPLICATE] "
                        f"{topic} "
                        f"partition={partition_id} "
                        f"offset={kafka_offset}"
                    )

                    # PostgreSQL already contains this
                    # Kafka record. The consumer offset
                    # can therefore safely advance.
                    self.consumer.commit(
                        message=message,
                        asynchronous=False,
                    )

                    processed += 1

                    if (
                        max_messages is not None
                        and processed >= max_messages
                    ):
                        break

                    continue

                # -----------------------------------------
                # Decode and update ExperimentEngine
                # -----------------------------------------

                payload = self._decode_json(
                    message
                )

                state = self._process_payload(
                    topic=topic,
                    payload=payload,
                )

                # -----------------------------------------
                # Durable PostgreSQL commit
                # -----------------------------------------
                #
                # Persist both the source event and the
                # latest experiment read model before
                # acknowledging the Kafka offset.
                # -----------------------------------------

                self.repository.persist_event_and_snapshot(
                    topic=topic,
                    partition_id=partition_id,
                    kafka_offset=kafka_offset,
                    event_key=self._message_key(
                        message
                    ),
                    payload=payload,
                    state=state,
                )

                # -----------------------------------------
                # Kafka offset acknowledgement
                # -----------------------------------------

                self.consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                processed += 1

                self._print_state(
                    topic,
                    state,
                )

                if (
                    max_messages is not None
                    and processed >= max_messages
                ):
                    break

        except KeyboardInterrupt:
            print()
            print(
                "Consumer stopped by user."
            )

        finally:
            self.consumer.close()

            state = self.engine.snapshot()

            print()
            print("=" * 60)
            print(
                "FINAL DURABLE EXPERIMENT STATE"
            )
            print("=" * 60)

            print(
                "Messages handled:",
                processed,
            )

            print(
                "Duplicate deliveries:",
                duplicates,
            )

            print(
                "Randomized CONTROL:",
                state.randomized_control,
            )

            print(
                "Randomized TREATMENT:",
                state.randomized_treatment,
            )

            print(
                "Total randomized:",
                state.total_randomized,
            )

            print(
                "Resolved:",
                state.resolved.total_resolved,
            )

            print(
                "Completed pairs:",
                state.inference.completed_pairs,
            )

            print(
                "Sequential evidence:",
                f"{state.inference.evidence:.6f}",
            )

            print(
                "Sequential boundary:",
                f"{state.inference.threshold:.1f}",
            )

            print(
                "Sequential decision:",
                state.inference.decision,
            )

            print(
                "SRM:",
                state.srm.status,
            )

            print(
                "High-value fraud:",
                state.guardrails.high_value_fraud.status,
            )

            print(
                "False decline:",
                state.guardrails.false_decline.status,
            )

            print(
                "Guardrail decision:",
                state.guardrails.decision,
            )

            print(
                "ENGINE DECISION:",
                state.decision,
            )

            self.repository.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:19092",
    )

    parser.add_argument(
        "--group-id",
        default=(
            "sequential-experiment-engine-"
            "postgres-v1"
        ),
    )

    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    stream_consumer = ExperimentStreamConsumer(
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        database_url=args.database_url,
    )

    stream_consumer.run(
        max_messages=args.max_messages
    )