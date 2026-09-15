import argparse
import json

from confluent_kafka import Consumer

from consumer.experiment_engine import (
    ExperimentEngine,
)
from producer.events import (
    FraudOutcomeEvent,
    TransactionEvent,
)


TRANSACTION_TOPIC = "transaction.events"
OUTCOME_TOPIC = "fraud.outcomes"


class ExperimentStreamConsumer:
    """
    Consumes transaction and delayed fraud events from
    Redpanda and applies them to ExperimentEngine.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:19092",
        group_id: str = (
            "sequential-experiment-engine-v1"
        ),
    ):

        self.engine = ExperimentEngine()

        self.consumer = Consumer(
            {
                "bootstrap.servers": (
                    bootstrap_servers
                ),
                "group.id": group_id,
                "auto.offset.reset": "earliest",

                # Commit only after the engine has
                # successfully processed an event.
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
    def _decode_json(
        message,
    ) -> dict:

        value = message.value()

        if value is None:
            raise ValueError(
                "Kafka message has no value."
            )

        return json.loads(
            value.decode("utf-8")
        )

    def _process_transaction(
        self,
        payload: dict,
    ):

        event = TransactionEvent(
            transaction_id=(
                payload["transaction_id"]
            ),
            card_id=payload["card_id"],
            experiment_id=(
                payload["experiment_id"]
            ),
            timestamp=payload["timestamp"],
            amount=float(
                payload["amount"]
            ),
            fraud_score=float(
                payload["fraud_score"]
            ),
        )

        return (
            self.engine.process_transaction(
                event
            )
        )

    def _process_outcome(
        self,
        payload: dict,
    ):

        event = FraudOutcomeEvent(
            transaction_id=(
                payload["transaction_id"]
            ),
            timestamp=payload["timestamp"],
            is_fraud=int(
                payload["is_fraud"]
            ),
        )

        return self.engine.process_outcome(
            event
        )

    @staticmethod
    def _print_state(
        topic: str,
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

        print(
            "REDPANDA EXPERIMENT CONSUMER"
        )
        print("-" * 60)

        print(
            "Listening to:"
        )
        print(
            f"  {TRANSACTION_TOPIC}"
        )
        print(
            f"  {OUTCOME_TOPIC}"
        )

        print("-" * 60)

        try:

            while True:

                message = (
                    self.consumer.poll(
                        timeout=1.0
                    )
                )

                if message is None:
                    continue

                if message.error():
                    raise RuntimeError(
                        str(message.error())
                    )

                topic = message.topic()

                payload = self._decode_json(
                    message
                )

                if topic == TRANSACTION_TOPIC:

                    state = (
                        self._process_transaction(
                            payload
                        )
                    )

                elif topic == OUTCOME_TOPIC:

                    state = (
                        self._process_outcome(
                            payload
                        )
                    )

                else:

                    raise ValueError(
                        f"Unexpected topic: {topic}"
                    )

                # Synchronous commit means the offset is
                # advanced only after successful engine
                # processing.
                self.consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                processed += 1

                self._print_state(
                    topic=topic,
                    state=state,
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
                "FINAL STREAMING EXPERIMENT STATE"
            )
            print("=" * 60)

            print(
                "Messages processed:",
                processed,
            )

            print(
                "Randomized:",
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
                "ENGINE DECISION:",
                state.decision,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bootstrap-servers",
        default="localhost:19092",
    )

    parser.add_argument(
        "--group-id",
        default=(
            "sequential-experiment-engine-v1"
        ),
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    stream_consumer = (
        ExperimentStreamConsumer(
            bootstrap_servers=(
                args.bootstrap_servers
            ),
            group_id=args.group_id,
        )
    )

    stream_consumer.run(
        max_messages=args.max_messages
    )