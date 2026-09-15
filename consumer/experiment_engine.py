from dataclasses import dataclass

from consumer.guardrails import (
    GuardrailSnapshot,
    OnlineGuardrails,
)
from consumer.operational_metrics import (
    OnlineOperationalMetrics,
    OperationalMetricsSnapshot,
)
from consumer.outcome_join import (
    OutcomeJoinState,
    ResolvedOutcome,
)
from consumer.pair_outcomes import (
    CompletedPair,
    PairOutcomeBuffer,
)
from consumer.pairing import PreOutcomePairing
from consumer.policy import evaluate_transaction
from consumer.sequential_inference import (
    StreamingInferenceSnapshot,
    StreamingSequentialInference,
)
from consumer.srm import (
    SRMResult,
    detect_srm,
)
from consumer.statistics import (
    ExperimentSnapshot,
    OnlineExperimentStats,
)
from producer.events import (
    FraudOutcomeEvent,
    TransactionEvent,
)


@dataclass(frozen=True)
class ExperimentEngineSnapshot:
    experiment_id: str

    randomized_control: int
    randomized_treatment: int
    total_randomized: int

    operational: OperationalMetricsSnapshot
    guardrails: GuardrailSnapshot
    resolved: ExperimentSnapshot
    inference: StreamingInferenceSnapshot
    srm: SRMResult

    decision: str


class ExperimentEngine:
    """
    In-memory orchestration layer for one sequential
    fraud-threshold experiment.

    Transaction path:

        TransactionEvent
            -> deterministic assignment / policy
            -> unique-card eligibility
            -> randomized-arm counts
            -> SRM monitoring
            -> operational metrics
            -> guardrail amount registration
            -> pre-outcome pairing
            -> delayed outcome join

    Outcome path:

        FraudOutcomeEvent
            -> delayed outcome join
            -> ResolvedOutcome
            -> online descriptive statistics
            -> delayed-label guardrails
            -> pair outcome buffer
            -> CompletedPair
            -> anytime-valid sequential inference

    Version 1 uses one eligible index transaction per
    card so that:

        randomization unit = inference unit = card

    SRM is evaluated using randomized unique-card
    counts, not resolved-outcome counts.

    Operational approval metrics update immediately
    when an eligible transaction is randomized and
    do not wait for delayed fraud outcomes.

    Guardrails use delayed fraud outcomes and are
    evaluated with anytime-valid confidence sequences.
    """

    def __init__(
        self,
        experiment_id: str = "fraud_threshold_v1",
        alpha: float = 0.05,
        prior_a: float = 100.0,
        prior_b: float = 100.0,
        srm_alpha: float = 0.001,
    ):

        if not experiment_id:
            raise ValueError(
                "experiment_id must be non-empty."
            )

        if not 0 < srm_alpha < 1:
            raise ValueError(
                "srm_alpha must be between 0 and 1."
            )

        self.experiment_id = experiment_id

        self.srm_alpha = srm_alpha

        # Current assignment policy uses a
        # 50/50 CONTROL/TREATMENT allocation.
        self.expected_treatment_fraction = 0.5

        self.join_state = OutcomeJoinState()

        self.pairing = PreOutcomePairing()

        self.pair_outcomes = PairOutcomeBuffer()

        self.stats = OnlineExperimentStats()

        self.operational_metrics = (
            OnlineOperationalMetrics()
        )

        self.guardrails = OnlineGuardrails(
            high_value_threshold=100.0,
            alpha=0.05,
        )

        self.inference = StreamingSequentialInference(
            alpha=alpha,
            prior_a=prior_a,
            prior_b=prior_b,
        )

        # SRM counts randomized unique cards at entry,
        # not resolved outcomes.
        self.randomized_control = 0
        self.randomized_treatment = 0

        # Version 1 permits exactly one eligible
        # index transaction per card.
        self.card_to_transaction_id: dict[
            str,
            str,
        ] = {}

        # Operational transaction-level idempotency.
        # This will later move to durable storage.
        self.seen_transactions: dict[
            str,
            TransactionEvent,
        ] = {}

        # Transactions rejected by the one-card /
        # one-index-transaction eligibility rule.
        self.ineligible_transaction_ids: set[
            str
        ] = set()

    def process_transaction(
        self,
        event: TransactionEvent,
    ) -> ExperimentEngineSnapshot:

        if event.experiment_id != self.experiment_id:
            raise ValueError(
                "Transaction belongs to experiment "
                f"{event.experiment_id}, expected "
                f"{self.experiment_id}."
            )

        # -------------------------------------------------
        # Transaction-level idempotency
        # -------------------------------------------------

        existing_event = self.seen_transactions.get(
            event.transaction_id
        )

        if existing_event is not None:

            if existing_event != event:
                raise ValueError(
                    "Conflicting duplicate transaction "
                    f"for {event.transaction_id}."
                )

            return self.snapshot()

        self.seen_transactions[
            event.transaction_id
        ] = event

        # -------------------------------------------------
        # One eligible index transaction per card
        # -------------------------------------------------

        existing_transaction_id = (
            self.card_to_transaction_id.get(
                event.card_id
            )
        )

        if existing_transaction_id is not None:

            self.ineligible_transaction_ids.add(
                event.transaction_id
            )

            # If an outcome arrived before this
            # transaction, remove it from pending
            # join state because this transaction is
            # not an experimental observation.
            self.join_state.pending_outcomes.pop(
                event.transaction_id,
                None,
            )

            return self.snapshot()

        # -------------------------------------------------
        # Deterministic experimental assignment
        # -------------------------------------------------

        decision = evaluate_transaction(
            event
        )

        self.card_to_transaction_id[
            event.card_id
        ] = event.transaction_id

        # -------------------------------------------------
        # Randomized-card counts for SRM
        # -------------------------------------------------

        if decision.arm == "CONTROL":

            self.randomized_control += 1

        elif decision.arm == "TREATMENT":

            self.randomized_treatment += 1

        else:

            raise ValueError(
                f"Unknown experiment arm: "
                f"{decision.arm}"
            )

        # -------------------------------------------------
        # Immediate operational business metrics
        # -------------------------------------------------

        # Approval metrics update at transaction arrival.
        # They do not wait for the delayed fraud label.
        self.operational_metrics.update(
            decision
        )

        # -------------------------------------------------
        # Guardrail transaction context
        # -------------------------------------------------

        # Amount is known immediately, while the fraud
        # label required by the guardrails arrives later.
        self.guardrails.register_transaction(
            transaction_id=event.transaction_id,
            amount=event.amount,
        )

        # -------------------------------------------------
        # Pre-outcome pairing
        # -------------------------------------------------

        pair = self.pairing.register(
            transaction_id=event.transaction_id,
            arm=decision.arm,
        )

        if pair is not None:

            completed_pair = (
                self.pair_outcomes.register_pair(
                    pair
                )
            )

            if completed_pair is not None:
                self._handle_completed_pair(
                    completed_pair
                )

        # -------------------------------------------------
        # Delayed outcome join
        # -------------------------------------------------

        resolved = (
            self.join_state.process_transaction(
                event
            )
        )

        # If the outcome arrived before the transaction,
        # the observation can resolve immediately here.
        if resolved is not None:
            self._handle_resolved(
                resolved
            )

        return self.snapshot()

    def process_outcome(
        self,
        outcome: FraudOutcomeEvent,
    ) -> ExperimentEngineSnapshot:

        # Outcomes belonging to transactions already
        # rejected by the one-card eligibility rule
        # must never contribute to experiment state.
        if (
            outcome.transaction_id
            in self.ineligible_transaction_ids
        ):
            return self.snapshot()

        resolved = self.join_state.process_outcome(
            outcome
        )

        if resolved is not None:
            self._handle_resolved(
                resolved
            )

        return self.snapshot()

    def _handle_resolved(
        self,
        resolved: ResolvedOutcome,
    ) -> None:

        if resolved.experiment_id != self.experiment_id:
            raise ValueError(
                "Resolved outcome belongs to "
                f"experiment "
                f"{resolved.experiment_id}, "
                f"expected {self.experiment_id}."
            )

        # -------------------------------------------------
        # Descriptive primary-metric statistics
        # -------------------------------------------------

        self.stats.update(
            resolved
        )

        # -------------------------------------------------
        # Delayed-label guardrails
        # -------------------------------------------------

        self.guardrails.update(
            resolved
        )

        # -------------------------------------------------
        # Pair outcome buffering
        # -------------------------------------------------

        completed_pair = (
            self.pair_outcomes.process_outcome(
                resolved
            )
        )

        if completed_pair is not None:
            self._handle_completed_pair(
                completed_pair
            )

    def _handle_completed_pair(
        self,
        completed_pair: CompletedPair,
    ) -> None:

        # Completed pairs update the anytime-valid
        # primary sequential test exactly once.
        self.inference.update(
            completed_pair
        )

    def _srm_result(
        self,
    ) -> SRMResult:

        return detect_srm(
            control_count=self.randomized_control,
            treatment_count=self.randomized_treatment,
            expected_treatment_fraction=(
                self.expected_treatment_fraction
            ),
            alpha=self.srm_alpha,
        )

    @staticmethod
    def _experiment_decision(
        srm: SRMResult,
        guardrails: GuardrailSnapshot,
        inference: StreamingInferenceSnapshot,
    ) -> str:

        # Randomization validity has highest priority.
        if srm.status == "INVALID_EXPERIMENT":
            return "INVALID_EXPERIMENT"

        # Safety/business guardrails override the
        # primary statistical result.
        if guardrails.decision == "STOP_GUARDRAIL":
            return "STOP_GUARDRAIL"

        return inference.decision

    def snapshot(
        self,
    ) -> ExperimentEngineSnapshot:

        operational_snapshot = (
            self.operational_metrics.snapshot()
        )

        guardrail_snapshot = (
            self.guardrails.snapshot()
        )

        resolved_snapshot = (
            self.stats.snapshot()
        )

        inference_snapshot = (
            self.inference.snapshot()
        )

        srm_result = self._srm_result()

        decision = self._experiment_decision(
            srm=srm_result,
            guardrails=guardrail_snapshot,
            inference=inference_snapshot,
        )

        return ExperimentEngineSnapshot(
            experiment_id=self.experiment_id,
            randomized_control=(
                self.randomized_control
            ),
            randomized_treatment=(
                self.randomized_treatment
            ),
            total_randomized=(
                self.randomized_control
                + self.randomized_treatment
            ),
            operational=operational_snapshot,
            guardrails=guardrail_snapshot,
            resolved=resolved_snapshot,
            inference=inference_snapshot,
            srm=srm_result,
            decision=decision,
        )


if __name__ == "__main__":

    engine = ExperimentEngine()

    # card_0101 deterministically maps to CONTROL.
    control_transaction = TransactionEvent(
        transaction_id="tx_control",
        card_id="card_0101",
        experiment_id="fraud_threshold_v1",
        timestamp="2026-09-15T20:00:00+00:00",
        amount=75.00,
        fraud_score=0.38,
    )

    # card_0042 deterministically maps to TREATMENT.
    treatment_transaction = TransactionEvent(
        transaction_id="tx_treatment",
        card_id="card_0042",
        experiment_id="fraud_threshold_v1",
        timestamp="2026-09-15T20:00:01+00:00",
        amount=75.00,
        fraud_score=0.38,
    )

    control_outcome = FraudOutcomeEvent(
        transaction_id="tx_control",
        timestamp="2026-09-15T20:30:00+00:00",
        is_fraud=1,
    )

    treatment_outcome = FraudOutcomeEvent(
        transaction_id="tx_treatment",
        timestamp="2026-09-15T20:31:00+00:00",
        is_fraud=1,
    )

    print("END-TO-END EXPERIMENT ENGINE")
    print("-" * 60)

    print()
    print("1. CONTROL TRANSACTION")
    print("-" * 60)

    state = engine.process_transaction(
        control_transaction
    )

    print(
        "Randomized:",
        state.total_randomized,
    )

    print(
        "Operational observations:",
        state.operational.total_transactions,
    )

    print(
        "Resolved:",
        state.resolved.total_resolved,
    )

    print(
        "SRM status:",
        state.srm.status,
    )

    print()
    print("2. TREATMENT TRANSACTION")
    print("-" * 60)

    state = engine.process_transaction(
        treatment_transaction
    )

    print(
        "Randomized:",
        state.total_randomized,
    )

    print(
        "Operational observations:",
        state.operational.total_transactions,
    )

    print(
        "Resolved:",
        state.resolved.total_resolved,
    )

    print(
        "SRM status:",
        state.srm.status,
    )

    print(
        "Pairs waiting for outcomes:",
        len(engine.pair_outcomes.pairs),
    )

    print()
    print("3. CONTROL FRAUD OUTCOME")
    print("-" * 60)

    state = engine.process_outcome(
        control_outcome
    )

    print(
        "Operational observations:",
        state.operational.total_transactions,
    )

    print(
        "Resolved:",
        state.resolved.total_resolved,
    )

    print(
        "Completed inference pairs:",
        state.inference.completed_pairs,
    )

    print()
    print("4. TREATMENT FRAUD OUTCOME")
    print("-" * 60)

    state = engine.process_outcome(
        treatment_outcome
    )

    print(
        "Operational observations:",
        state.operational.total_transactions,
    )

    print(
        "Resolved:",
        state.resolved.total_resolved,
    )

    print(
        "Completed inference pairs:",
        state.inference.completed_pairs,
    )

    print()
    print("FINAL EXPERIMENT STATE")
    print("-" * 60)

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

    print()

    print(
        "Control approval rate:",
        f"{state.operational.control_approval_rate:.2%}",
    )

    print(
        "Treatment approval rate:",
        f"{state.operational.treatment_approval_rate:.2%}",
    )

    print(
        "Approval-rate difference:",
        f"{state.operational.approval_rate_difference * 100:.2f} pp",
    )

    print()

    print(
        "High-value fraud guardrail:",
        state.guardrails.high_value_fraud.status,
    )

    print(
        "False-decline guardrail:",
        state.guardrails.false_decline.status,
    )

    print(
        "Guardrail decision:",
        state.guardrails.decision,
    )

    print()

    print(
        "SRM treatment fraction:",
        f"{state.srm.observed_treatment_fraction:.2%}",
    )

    print(
        "SRM evidence:",
        f"{state.srm.evidence:.6f}",
    )

    print(
        "SRM boundary:",
        f"{state.srm.threshold:.0f}",
    )

    print(
        "SRM status:",
        state.srm.status,
    )

    print()

    print(
        "Resolved CONTROL:",
        state.resolved.control_n,
    )

    print(
        "Resolved TREATMENT:",
        state.resolved.treatment_n,
    )

    print(
        "Control leakage rate:",
        f"{state.resolved.control_rate:.2%}",
    )

    print(
        "Treatment leakage rate:",
        f"{state.resolved.treatment_rate:.2%}",
    )

    print(
        "Leakage-rate difference:",
        f"{state.resolved.difference * 100:.2f} pp",
    )

    print()

    print(
        "Completed pairs:",
        state.inference.completed_pairs,
    )

    print(
        "Positive discordants:",
        state.inference.positive_discordants,
    )

    print(
        "Negative discordants:",
        state.inference.negative_discordants,
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

    print()

    print(
        "ENGINE DECISION:",
        state.decision,
    )