from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln

from consumer.outcome_join import ResolvedOutcome


@dataclass(frozen=True)
class GuardrailMetricSnapshot:
    control_n: int
    control_events: int
    treatment_n: int
    treatment_events: int

    control_rate: float
    treatment_rate: float
    difference: float

    difference_lower: float
    difference_upper: float

    status: str


@dataclass(frozen=True)
class GuardrailSnapshot:
    high_value_fraud: GuardrailMetricSnapshot
    false_decline: GuardrailMetricSnapshot
    decision: str


@dataclass
class _BinaryCounts:
    control_n: int = 0
    control_events: int = 0
    treatment_n: int = 0
    treatment_events: int = 0


def _bernoulli_log_evidence(
    n: int,
    x: int,
    p0: float,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> float:

    if n == 0:
        return 0.0

    log_mixture = (
        betaln(
            prior_a + x,
            prior_b + n - x,
        )
        - betaln(
            prior_a,
            prior_b,
        )
    )

    if p0 == 0.0:

        if x > 0:
            return float("inf")

        log_null = 0.0

    elif p0 == 1.0:

        if x < n:
            return float("inf")

        log_null = 0.0

    else:

        log_null = (
            x * np.log(p0)
            + (n - x) * np.log1p(-p0)
        )

    return float(
        log_mixture - log_null
    )


def bernoulli_confidence_sequence(
    n: int,
    x: int,
    alpha: float,
) -> tuple[float, float]:
    """
    Anytime-valid Bernoulli confidence sequence obtained
    by inverting a Jeffreys Beta(0.5, 0.5) mixture
    e-process.

    Returns a confidence interval for the Bernoulli
    probability after n observations.
    """

    if n < 0:
        raise ValueError(
            "n must be non-negative."
        )

    if x < 0 or x > n:
        raise ValueError(
            "x must satisfy 0 <= x <= n."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    if n == 0:
        return 0.0, 1.0

    log_boundary = np.log(
        1.0 / alpha
    )

    observed_rate = x / n

    def objective(p0: float) -> float:

        return (
            _bernoulli_log_evidence(
                n=n,
                x=x,
                p0=p0,
            )
            - log_boundary
        )

    epsilon = 1e-12

    if x == 0:

        lower = 0.0

    else:

        lower = brentq(
            objective,
            epsilon,
            observed_rate,
        )

    if x == n:

        upper = 1.0

    else:

        upper = brentq(
            objective,
            observed_rate,
            1.0 - epsilon,
        )

    return (
        float(lower),
        float(upper),
    )


class OnlineGuardrails:
    """
    Maintains two delayed-label guardrails:

    1. High-value fraud leakage
       amount >= high_value_threshold
       AND approved
       AND fraudulent

    2. False decline
       declined AND non-fraudulent

    Both are Bernoulli outcomes per eligible resolved
    experimental unit.

    Guardrail confidence is allocated across:

        2 guardrails x 2 experiment arms

    using a union bound.

    A guardrail breaches only when the anytime-valid
    lower bound for:

        treatment rate - control rate

    is strictly greater than zero.
    """

    def __init__(
        self,
        high_value_threshold: float = 100.0,
        alpha: float = 0.05,
    ):

        if high_value_threshold < 0:
            raise ValueError(
                "high_value_threshold must be "
                "non-negative."
            )

        if not 0 < alpha < 1:
            raise ValueError(
                "alpha must be between 0 and 1."
            )

        self.high_value_threshold = (
            high_value_threshold
        )

        self.alpha = alpha

        # Two guardrails and two arms.
        self.arm_alpha = alpha / 4.0

        self.high_value_fraud = (
            _BinaryCounts()
        )

        self.false_decline = (
            _BinaryCounts()
        )

        # Amount is known at transaction arrival,
        # whereas the fraud label arrives later.
        self.pending_amounts: dict[
            str,
            float,
        ] = {}

    def register_transaction(
        self,
        transaction_id: str,
        amount: float,
    ) -> None:

        if not transaction_id:
            raise ValueError(
                "transaction_id must be non-empty."
            )

        if amount < 0:
            raise ValueError(
                "amount must be non-negative."
            )

        existing = self.pending_amounts.get(
            transaction_id
        )

        if existing is not None:

            if existing != amount:
                raise ValueError(
                    "Conflicting amount for "
                    f"{transaction_id}."
                )

            return

        self.pending_amounts[
            transaction_id
        ] = amount

    @staticmethod
    def _update_counts(
        counts: _BinaryCounts,
        arm: str,
        event_value: int,
    ) -> None:

        if event_value not in (0, 1):
            raise ValueError(
                "event_value must be 0 or 1."
            )

        if arm == "CONTROL":

            counts.control_n += 1
            counts.control_events += (
                event_value
            )

        elif arm == "TREATMENT":

            counts.treatment_n += 1
            counts.treatment_events += (
                event_value
            )

        else:

            raise ValueError(
                f"Unknown experiment arm: {arm}"
            )

    def update(
        self,
        outcome: ResolvedOutcome,
    ) -> None:

        if (
            outcome.transaction_id
            not in self.pending_amounts
        ):
            raise ValueError(
                "No registered transaction amount "
                f"for {outcome.transaction_id}."
            )

        amount = self.pending_amounts.pop(
            outcome.transaction_id
        )

        high_value_fraud = int(
            amount >= self.high_value_threshold
            and outcome.fraud_leakage == 1
        )

        false_decline = int(
            not outcome.approved
            and outcome.is_fraud == 0
        )

        self._update_counts(
            counts=self.high_value_fraud,
            arm=outcome.arm,
            event_value=high_value_fraud,
        )

        self._update_counts(
            counts=self.false_decline,
            arm=outcome.arm,
            event_value=false_decline,
        )

    def _metric_snapshot(
        self,
        counts: _BinaryCounts,
    ) -> GuardrailMetricSnapshot:

        if counts.control_n == 0:
            control_rate = 0.0
        else:
            control_rate = (
                counts.control_events
                / counts.control_n
            )

        if counts.treatment_n == 0:
            treatment_rate = 0.0
        else:
            treatment_rate = (
                counts.treatment_events
                / counts.treatment_n
            )

        control_lower, control_upper = (
            bernoulli_confidence_sequence(
                n=counts.control_n,
                x=counts.control_events,
                alpha=self.arm_alpha,
            )
        )

        treatment_lower, treatment_upper = (
            bernoulli_confidence_sequence(
                n=counts.treatment_n,
                x=counts.treatment_events,
                alpha=self.arm_alpha,
            )
        )

        difference = (
            treatment_rate
            - control_rate
        )

        difference_lower = (
            treatment_lower
            - control_upper
        )

        difference_upper = (
            treatment_upper
            - control_lower
        )

        if difference_lower > 0.0:
            status = "BREACH"
        else:
            status = "MONITORING"

        return GuardrailMetricSnapshot(
            control_n=counts.control_n,
            control_events=(
                counts.control_events
            ),
            treatment_n=counts.treatment_n,
            treatment_events=(
                counts.treatment_events
            ),
            control_rate=control_rate,
            treatment_rate=treatment_rate,
            difference=difference,
            difference_lower=(
                difference_lower
            ),
            difference_upper=(
                difference_upper
            ),
            status=status,
        )

    def snapshot(
        self,
    ) -> GuardrailSnapshot:

        high_value_snapshot = (
            self._metric_snapshot(
                self.high_value_fraud
            )
        )

        false_decline_snapshot = (
            self._metric_snapshot(
                self.false_decline
            )
        )

        if (
            high_value_snapshot.status
            == "BREACH"
            or false_decline_snapshot.status
            == "BREACH"
        ):

            decision = "STOP_GUARDRAIL"

        else:

            decision = "CONTINUE"

        return GuardrailSnapshot(
            high_value_fraud=(
                high_value_snapshot
            ),
            false_decline=(
                false_decline_snapshot
            ),
            decision=decision,
        )
