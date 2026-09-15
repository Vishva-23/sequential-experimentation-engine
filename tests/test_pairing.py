import pytest

from consumer.pairing import PreOutcomePairing


def test_control_then_treatment_forms_pair():

    pairing = PreOutcomePairing()

    first = pairing.register(
        transaction_id="tx_c1",
        arm="CONTROL",
    )

    second = pairing.register(
        transaction_id="tx_t1",
        arm="TREATMENT",
    )

    assert first is None
    assert second is not None

    assert second.pair_id == 1
    assert second.control_transaction_id == "tx_c1"
    assert second.treatment_transaction_id == "tx_t1"


def test_treatment_then_control_forms_pair():

    pairing = PreOutcomePairing()

    first = pairing.register(
        transaction_id="tx_t1",
        arm="TREATMENT",
    )

    second = pairing.register(
        transaction_id="tx_c1",
        arm="CONTROL",
    )

    assert first is None
    assert second is not None

    assert second.control_transaction_id == "tx_c1"
    assert second.treatment_transaction_id == "tx_t1"


def test_fifo_pairing_with_multiple_transactions():

    pairing = PreOutcomePairing()

    pairing.register("tx_c1", "CONTROL")
    pairing.register("tx_c2", "CONTROL")

    pair_1 = pairing.register(
        "tx_t1",
        "TREATMENT",
    )

    pair_2 = pairing.register(
        "tx_t2",
        "TREATMENT",
    )

    assert pair_1 is not None
    assert pair_2 is not None

    assert pair_1.control_transaction_id == "tx_c1"
    assert pair_1.treatment_transaction_id == "tx_t1"

    assert pair_2.control_transaction_id == "tx_c2"
    assert pair_2.treatment_transaction_id == "tx_t2"


def test_pair_lookup_works_for_both_members():

    pairing = PreOutcomePairing()

    pairing.register("tx_c1", "CONTROL")
    pairing.register("tx_t1", "TREATMENT")

    assert pairing.get_pair_id("tx_c1") == 1
    assert pairing.get_pair_id("tx_t1") == 1


def test_duplicate_transaction_is_ignored():

    pairing = PreOutcomePairing()

    first = pairing.register(
        "tx_c1",
        "CONTROL",
    )

    duplicate = pairing.register(
        "tx_c1",
        "CONTROL",
    )

    assert first is None
    assert duplicate is None

    assert len(pairing.unpaired_control) == 1


def test_invalid_arm_is_rejected():

    pairing = PreOutcomePairing()

    with pytest.raises(ValueError):
        pairing.register(
            transaction_id="tx_bad",
            arm="INVALID",
        )


def test_empty_transaction_id_is_rejected():

    pairing = PreOutcomePairing()

    with pytest.raises(ValueError):
        pairing.register(
            transaction_id="",
            arm="CONTROL",
        )