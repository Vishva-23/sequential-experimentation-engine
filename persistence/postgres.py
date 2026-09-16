import json
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS processed_events (
    id BIGSERIAL PRIMARY KEY,

    topic TEXT NOT NULL,
    partition_id INTEGER NOT NULL,
    kafka_offset BIGINT NOT NULL,

    event_key TEXT,
    payload JSONB NOT NULL,

    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (topic, partition_id, kafka_offset)
);


CREATE TABLE IF NOT EXISTS experiment_state (
    experiment_id TEXT PRIMARY KEY,

    randomized_control INTEGER NOT NULL,
    randomized_treatment INTEGER NOT NULL,
    total_randomized INTEGER NOT NULL,

    total_resolved INTEGER NOT NULL,
    completed_pairs INTEGER NOT NULL,

    sequential_evidence DOUBLE PRECISION NOT NULL,
    sequential_boundary DOUBLE PRECISION NOT NULL,

    sequential_decision TEXT NOT NULL,
    srm_status TEXT NOT NULL,
    guardrail_decision TEXT NOT NULL,
    engine_decision TEXT NOT NULL,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS arm_state (
    experiment_id TEXT NOT NULL,
    arm TEXT NOT NULL,

    randomized_n INTEGER NOT NULL,

    operational_n INTEGER NOT NULL,
    approved_n INTEGER NOT NULL,
    approval_rate DOUBLE PRECISION NOT NULL,

    resolved_n INTEGER NOT NULL,
    fraud_leakage_n INTEGER NOT NULL,
    fraud_leakage_rate DOUBLE PRECISION NOT NULL,

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (experiment_id, arm)
);


CREATE TABLE IF NOT EXISTS decisions (
    id BIGSERIAL PRIMARY KEY,

    experiment_id TEXT NOT NULL,

    engine_decision TEXT NOT NULL,
    sequential_decision TEXT NOT NULL,
    srm_status TEXT NOT NULL,
    guardrail_decision TEXT NOT NULL,

    sequential_evidence DOUBLE PRECISION NOT NULL,
    sequential_boundary DOUBLE PRECISION NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE INDEX IF NOT EXISTS idx_processed_events_order
ON processed_events(id);

CREATE INDEX IF NOT EXISTS idx_decisions_experiment
ON decisions(experiment_id, id DESC);
"""


@dataclass(frozen=True)
class StoredEvent:
    topic: str
    partition_id: int
    kafka_offset: int
    event_key: str | None
    payload: dict[str, Any]


class PostgresExperimentRepository:

    def __init__(
        self,
        database_url: str,
    ):
        self.database_url = database_url

        # Autocommit prevents ordinary SELECT queries from
        # leaving an implicit outer transaction open.
        #
        # Durable multi-table writes still use explicit
        # connection.transaction() blocks below.
        self.connection = psycopg.connect(
            database_url,
            autocommit=True,
        )

    def close(self) -> None:
        self.connection.close()

    def ensure_schema(self) -> None:
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(SCHEMA_SQL)

    def event_exists(
        self,
        topic: str,
        partition_id: int,
        kafka_offset: int,
    ) -> bool:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM processed_events
                WHERE topic = %s
                  AND partition_id = %s
                  AND kafka_offset = %s
                LIMIT 1
                """,
                (
                    topic,
                    partition_id,
                    kafka_offset,
                ),
            )

            return cursor.fetchone() is not None

    def load_events(
        self,
    ) -> list[StoredEvent]:
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    topic,
                    partition_id,
                    kafka_offset,
                    event_key,
                    payload
                FROM processed_events
                ORDER BY id
                """
            )

            rows = cursor.fetchall()

        events = []

        for row in rows:
            payload = row[4]

            if isinstance(payload, str):
                payload = json.loads(payload)

            events.append(
                StoredEvent(
                    topic=row[0],
                    partition_id=row[1],
                    kafka_offset=row[2],
                    event_key=row[3],
                    payload=payload,
                )
            )

        return events

    def persist_event_and_snapshot(
        self,
        topic: str,
        partition_id: int,
        kafka_offset: int,
        event_key: str | None,
        payload: dict[str, Any],
        state,
    ) -> None:
        # One explicit PostgreSQL transaction contains:
        #
        #   1. processed Kafka event
        #   2. experiment snapshot
        #   3. control arm snapshot
        #   4. treatment arm snapshot
        #   5. decision transition, if changed
        #
        # Either all of these writes commit or none do.
        with self.connection.transaction():
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO processed_events (
                        topic,
                        partition_id,
                        kafka_offset,
                        event_key,
                        payload
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (
                        topic,
                        partition_id,
                        kafka_offset
                    )
                    DO NOTHING
                    """,
                    (
                        topic,
                        partition_id,
                        kafka_offset,
                        event_key,
                        Jsonb(payload),
                    ),
                )

                self._upsert_experiment_state(
                    cursor,
                    state,
                )

                self._upsert_arm_state(
                    cursor,
                    state,
                    arm="CONTROL",
                )

                self._upsert_arm_state(
                    cursor,
                    state,
                    arm="TREATMENT",
                )

                self._record_decision_if_changed(
                    cursor,
                    state,
                )

    @staticmethod
    def _upsert_experiment_state(
        cursor,
        state,
    ) -> None:
        cursor.execute(
            """
            INSERT INTO experiment_state (
                experiment_id,

                randomized_control,
                randomized_treatment,
                total_randomized,

                total_resolved,
                completed_pairs,

                sequential_evidence,
                sequential_boundary,

                sequential_decision,
                srm_status,
                guardrail_decision,
                engine_decision,

                updated_at
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                NOW()
            )

            ON CONFLICT (experiment_id)
            DO UPDATE SET
                randomized_control =
                    EXCLUDED.randomized_control,

                randomized_treatment =
                    EXCLUDED.randomized_treatment,

                total_randomized =
                    EXCLUDED.total_randomized,

                total_resolved =
                    EXCLUDED.total_resolved,

                completed_pairs =
                    EXCLUDED.completed_pairs,

                sequential_evidence =
                    EXCLUDED.sequential_evidence,

                sequential_boundary =
                    EXCLUDED.sequential_boundary,

                sequential_decision =
                    EXCLUDED.sequential_decision,

                srm_status =
                    EXCLUDED.srm_status,

                guardrail_decision =
                    EXCLUDED.guardrail_decision,

                engine_decision =
                    EXCLUDED.engine_decision,

                updated_at = NOW()
            """,
            (
                state.experiment_id,

                state.randomized_control,
                state.randomized_treatment,
                state.total_randomized,

                state.resolved.total_resolved,
                state.inference.completed_pairs,

                state.inference.evidence,
                state.inference.threshold,

                state.inference.decision,
                state.srm.status,
                state.guardrails.decision,
                state.decision,
            ),
        )

    @staticmethod
    def _upsert_arm_state(
        cursor,
        state,
        arm: str,
    ) -> None:
        if arm == "CONTROL":
            randomized_n = (
                state.randomized_control
            )

            operational_n = (
                state.operational.control_n
            )

            approved_n = (
                state.operational.control_approved
            )

            approval_rate = (
                state.operational.control_approval_rate
            )

            resolved_n = (
                state.resolved.control_n
            )

            fraud_leakage_n = (
                state.resolved.control_fraud_leakage
            )

            fraud_leakage_rate = (
                state.resolved.control_rate
            )

        elif arm == "TREATMENT":
            randomized_n = (
                state.randomized_treatment
            )

            operational_n = (
                state.operational.treatment_n
            )

            approved_n = (
                state.operational.treatment_approved
            )

            approval_rate = (
                state.operational.treatment_approval_rate
            )

            resolved_n = (
                state.resolved.treatment_n
            )

            fraud_leakage_n = (
                state.resolved.treatment_fraud_leakage
            )

            fraud_leakage_rate = (
                state.resolved.treatment_rate
            )

        else:
            raise ValueError(
                f"Unknown arm: {arm}"
            )

        cursor.execute(
            """
            INSERT INTO arm_state (
                experiment_id,
                arm,

                randomized_n,

                operational_n,
                approved_n,
                approval_rate,

                resolved_n,
                fraud_leakage_n,
                fraud_leakage_rate,

                updated_at
            )
            VALUES (
                %s, %s,
                %s,
                %s, %s, %s,
                %s, %s, %s,
                NOW()
            )

            ON CONFLICT (experiment_id, arm)
            DO UPDATE SET
                randomized_n =
                    EXCLUDED.randomized_n,

                operational_n =
                    EXCLUDED.operational_n,

                approved_n =
                    EXCLUDED.approved_n,

                approval_rate =
                    EXCLUDED.approval_rate,

                resolved_n =
                    EXCLUDED.resolved_n,

                fraud_leakage_n =
                    EXCLUDED.fraud_leakage_n,

                fraud_leakage_rate =
                    EXCLUDED.fraud_leakage_rate,

                updated_at = NOW()
            """,
            (
                state.experiment_id,
                arm,

                randomized_n,

                operational_n,
                approved_n,
                approval_rate,

                resolved_n,
                fraud_leakage_n,
                fraud_leakage_rate,
            ),
        )

    @staticmethod
    def _record_decision_if_changed(
        cursor,
        state,
    ) -> None:
        cursor.execute(
            """
            SELECT engine_decision
            FROM decisions
            WHERE experiment_id = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                state.experiment_id,
            ),
        )

        row = cursor.fetchone()

        previous_decision = (
            row[0]
            if row is not None
            else None
        )

        if previous_decision == state.decision:
            return

        cursor.execute(
            """
            INSERT INTO decisions (
                experiment_id,

                engine_decision,
                sequential_decision,
                srm_status,
                guardrail_decision,

                sequential_evidence,
                sequential_boundary
            )
            VALUES (
                %s,
                %s, %s, %s, %s,
                %s, %s
            )
            """,
            (
                state.experiment_id,

                state.decision,
                state.inference.decision,
                state.srm.status,
                state.guardrails.decision,

                state.inference.evidence,
                state.inference.threshold,
            ),
        )