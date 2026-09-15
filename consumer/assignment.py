from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class AssignmentResult:
    experiment_id: str
    card_id: str
    arm: str
    fraud_threshold: float


def assignment_value(
    experiment_id: str,
    card_id: str,
) -> float:
    """
    Deterministically maps an experiment/card pair
    to a value in [0, 1).
    """

    if not experiment_id:
        raise ValueError(
            "experiment_id cannot be empty."
        )

    if not card_id:
        raise ValueError(
            "card_id cannot be empty."
        )

    key = (
        f"{experiment_id}:{card_id}"
    ).encode("utf-8")

    digest = hashlib.sha256(key).digest()

    # Use the first 8 bytes as an unsigned
    # 64-bit integer.
    integer_value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return integer_value / (2 ** 64)


def assign_card(
    experiment_id: str,
    card_id: str,
    treatment_allocation: float = 0.5,
    control_threshold: float = 0.35,
    treatment_threshold: float = 0.42,
) -> AssignmentResult:

    if not 0 < treatment_allocation < 1:
        raise ValueError(
            "treatment_allocation must be "
            "between 0 and 1."
        )

    value = assignment_value(
        experiment_id=experiment_id,
        card_id=card_id,
    )

    if value < treatment_allocation:
        arm = "TREATMENT"
        fraud_threshold = treatment_threshold
    else:
        arm = "CONTROL"
        fraud_threshold = control_threshold

    return AssignmentResult(
        experiment_id=experiment_id,
        card_id=card_id,
        arm=arm,
        fraud_threshold=fraud_threshold,
    )


if __name__ == "__main__":

    experiment_id = "fraud_threshold_v1"

    card_ids = [
        "card_0042",
        "card_0042",
        "card_0100",
        "card_0101",
        "card_0102",
    ]

    print("DETERMINISTIC ASSIGNMENT")
    print("-" * 48)

    for card_id in card_ids:

        result = assign_card(
            experiment_id=experiment_id,
            card_id=card_id,
        )

        print(
            f"{card_id:12} -> "
            f"{result.arm:9} | "
            f"threshold={result.fraud_threshold:.2f}"
        )