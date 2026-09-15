from dataclasses import dataclass

from consumer.outcome_join import ResolvedOutcome
from consumer.pairing import PairAssignment


@dataclass(frozen=True)
class CompletedPair:
    pair_id: int

    control_transaction_id: str
    treatment_transaction_id: str

    control_y: int
    treatment_y: int

    @property
    def is_discordant(self) -> bool:
        return self.control_y != self.treatment_y

    @property
    def direction(self) -> str:
        if self.control_y == 0 and self.treatment_y == 1:
            return "POSITIVE"

        if self.control_y == 1 and self.treatment_y == 0:
            return "NEGATIVE"

        return "CONCORDANT"


class PairOutcomeBuffer:
    """
    Combines predetermined pair assignments with
    delayed resolved outcomes.

    Pair membership is determined before fraud labels
    are used. Outcome arrival order cannot change which
    control and treatment observations are paired.
    """

    def __init__(self):

        self.pairs: dict[int, PairAssignment] = {}

        self.transaction_to_pair: dict[str, int] = {}

        self.resolved_outcomes: dict[
            str,
            ResolvedOutcome,
        ] = {}

        self.completed_pair_ids: set[int] = set()

    def register_pair(
        self,
        pair: PairAssignment,
    ) -> CompletedPair | None:

        if pair.pair_id in self.completed_pair_ids:
            return None

        existing = self.pairs.get(
            pair.pair_id
        )

        if existing is not None:
            if existing != pair:
                raise ValueError(
                    f"Conflicting pair definition "
                    f"for pair {pair.pair_id}."
                )

            return None

        self._validate_pair_transactions(pair)

        self.pairs[pair.pair_id] = pair

        self.transaction_to_pair[
            pair.control_transaction_id
        ] = pair.pair_id

        self.transaction_to_pair[
            pair.treatment_transaction_id
        ] = pair.pair_id

        return self._try_complete(
            pair.pair_id
        )

    def process_outcome(
        self,
        outcome: ResolvedOutcome,
    ) -> CompletedPair | None:

        transaction_id = outcome.transaction_id

        pair_id = self.transaction_to_pair.get(
            transaction_id
        )

        # If this transaction belongs to a pair that has
        # already completed, this is duplicate delivery.
        if (
            pair_id is not None
            and pair_id in self.completed_pair_ids
        ):
            return None

        existing = self.resolved_outcomes.get(
            transaction_id
        )

        if existing is not None:
            if existing != outcome:
                raise ValueError(
                    "Conflicting resolved outcome "
                    f"for {transaction_id}."
                )

            return None

        self.resolved_outcomes[
            transaction_id
        ] = outcome

        # The outcome may arrive before its pair
        # assignment has been registered.
        if pair_id is None:
            return None

        return self._try_complete(
            pair_id
        )

    def _validate_pair_transactions(
        self,
        pair: PairAssignment,
    ) -> None:

        if (
            pair.control_transaction_id
            == pair.treatment_transaction_id
        ):
            raise ValueError(
                "A transaction cannot be paired "
                "with itself."
            )

        for transaction_id in (
            pair.control_transaction_id,
            pair.treatment_transaction_id,
        ):

            existing_pair_id = (
                self.transaction_to_pair.get(
                    transaction_id
                )
            )

            if (
                existing_pair_id is not None
                and existing_pair_id != pair.pair_id
            ):
                raise ValueError(
                    f"{transaction_id} is already "
                    f"assigned to pair "
                    f"{existing_pair_id}."
                )

    def _try_complete(
        self,
        pair_id: int,
    ) -> CompletedPair | None:

        if pair_id in self.completed_pair_ids:
            return None

        pair = self.pairs.get(
            pair_id
        )

        if pair is None:
            return None

        control = self.resolved_outcomes.get(
            pair.control_transaction_id
        )

        treatment = self.resolved_outcomes.get(
            pair.treatment_transaction_id
        )

        # Both delayed outcomes must be available
        # before the pair contributes to inference.
        if control is None or treatment is None:
            return None

        if control.arm != "CONTROL":
            raise ValueError(
                "Control transaction resolved with "
                f"arm {control.arm}."
            )

        if treatment.arm != "TREATMENT":
            raise ValueError(
                "Treatment transaction resolved with "
                f"arm {treatment.arm}."
            )

        completed = CompletedPair(
            pair_id=pair_id,
            control_transaction_id=(
                pair.control_transaction_id
            ),
            treatment_transaction_id=(
                pair.treatment_transaction_id
            ),
            control_y=control.fraud_leakage,
            treatment_y=treatment.fraud_leakage,
        )

        self.completed_pair_ids.add(
            pair_id
        )

        # These outcomes are no longer needed by
        # the in-memory pairing buffer.
        del self.resolved_outcomes[
            pair.control_transaction_id
        ]

        del self.resolved_outcomes[
            pair.treatment_transaction_id
        ]

        return completed


if __name__ == "__main__":

    buffer = PairOutcomeBuffer()

    control = ResolvedOutcome(
        transaction_id="tx_c1",
        card_id="card_c1",
        experiment_id="fraud_threshold_v1",
        arm="CONTROL",
        approved=False,
        is_fraud=0,
        fraud_leakage=0,
    )

    treatment = ResolvedOutcome(
        transaction_id="tx_t1",
        card_id="card_t1",
        experiment_id="fraud_threshold_v1",
        arm="TREATMENT",
        approved=True,
        is_fraud=1,
        fraud_leakage=1,
    )

    pair = PairAssignment(
        pair_id=1,
        control_transaction_id="tx_c1",
        treatment_transaction_id="tx_t1",
    )

    print("PAIR OUTCOME BUFFER")
    print("-" * 50)

    print(
        "Control outcome before pair:",
        buffer.process_outcome(control),
    )

    print(
        "Register pair:",
        buffer.register_pair(pair),
    )

    completed = buffer.process_outcome(
        treatment
    )

    print(
        "Treatment outcome:",
        completed,
    )

    print()
    print("COMPLETED PAIR")
    print("-" * 50)

    print(
        "Control Y:",
        completed.control_y,
    )

    print(
        "Treatment Y:",
        completed.treatment_y,
    )

    print(
        "Discordant:",
        completed.is_discordant,
    )

    print(
        "Direction:",
        completed.direction,
    )