from collections import deque
from dataclasses import dataclass
@dataclass(frozen=True)
class PairAssignment:
    pair_id: int
    control_transaction_id: str
    treatment_transaction_id: str


class PreOutcomePairing:
    """
    Forms CONTROL/TREATMENT transaction pairs before
    fraud outcomes are observed.

    Pairing depends only on transaction arrival and arm,
    never on the fraud outcome or outcome arrival time.
    """

    def __init__(self):

        self.unpaired_control: deque[str] = deque()
        self.unpaired_treatment: deque[str] = deque()
        self.transaction_to_pair: dict[str, int] = {}

        self.pairs: dict[int, PairAssignment] = {}

        self.next_pair_id = 1

        self.seen_transaction_ids: set[str] = set()

    def register(
        self,
        transaction_id: str,
        arm: str,
    ) -> PairAssignment | None:

        if not transaction_id:
            raise ValueError(
                "transaction_id must be non-empty."
            )

        if arm not in ("CONTROL", "TREATMENT"):
            raise ValueError(
                f"Unknown experiment arm: {arm}"
            )

        if transaction_id in self.seen_transaction_ids:
            return None

        self.seen_transaction_ids.add(
            transaction_id
        )

        if arm == "CONTROL":

            if self.unpaired_treatment:

                treatment_id = (
                    self.unpaired_treatment.popleft()
                )

                return self._create_pair(
                    control_transaction_id=transaction_id,
                    treatment_transaction_id=treatment_id,
                )

            self.unpaired_control.append(
                transaction_id
            )

            return None

        if self.unpaired_control:

            control_id = (
                self.unpaired_control.popleft()
            )

            return self._create_pair(
                control_transaction_id=control_id,
                treatment_transaction_id=transaction_id,
            )

        self.unpaired_treatment.append(
            transaction_id
        )

        return None

    def _create_pair(
        self,
        control_transaction_id: str,
        treatment_transaction_id: str,
    ) -> PairAssignment:

        pair_id = self.next_pair_id
        self.next_pair_id += 1

        pair = PairAssignment(
            pair_id=pair_id,
            control_transaction_id=(
                control_transaction_id
            ),
            treatment_transaction_id=(
                treatment_transaction_id
            ),
        )

        self.pairs[pair_id] = pair

        self.transaction_to_pair[
            control_transaction_id
        ] = pair_id

        self.transaction_to_pair[
            treatment_transaction_id
        ] = pair_id

        return pair

    def get_pair_id(
        self,
        transaction_id: str,
    ) -> int | None:

        return self.transaction_to_pair.get(
            transaction_id
        )


if __name__ == "__main__":

    pairing = PreOutcomePairing()

    arrivals = [
        ("tx_c1", "CONTROL"),
        ("tx_c2", "CONTROL"),
        ("tx_t1", "TREATMENT"),
        ("tx_t2", "TREATMENT"),
    ]

    print("PRE-OUTCOME PAIRING")
    print("-" * 50)

    for transaction_id, arm in arrivals:

        pair = pairing.register(
            transaction_id=transaction_id,
            arm=arm,
        )

        print(
            f"{transaction_id:6} "
            f"{arm:9} "
            f"-> {pair}"
        )

    print()
    print("PAIR LOOKUP")
    print("-" * 50)

    for transaction_id, _ in arrivals:

        print(
            transaction_id,
            "-> pair",
            pairing.get_pair_id(
                transaction_id
            ),
        )