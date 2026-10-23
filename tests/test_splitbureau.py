"""Tests for splitbureau: PSI alignment, split-learning training, cut-layer leakage and API."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from splitbureau.api import app
from splitbureau.data import build_aligned_dataset_from_registries, generate_unaligned_registries
from splitbureau.psi import align_three_parties, align_two_parties
from splitbureau.splitnn import CutLayerDefenseConfig, SplitCreditNet, distance_correlation, train_bank_only_baseline


def test_two_party_psi_finds_exactly_the_common_ids() -> None:
    a = ["p1", "P2", "only_a", "p3"]
    b = ["x", " p3", "p2", "only_b", "P1"]
    res = align_two_parties(a, b)
    pairs = {(a[i].strip().upper(), b[j].strip().upper()) for i, j in zip(res.aligned_indices_a, res.aligned_indices_b)}
    assert pairs == {("P1", "P1"), ("P2", "P2"), ("P3", "P3")}
    assert res.exact_recovery_rate == 1.0


def test_three_party_psi_on_generated_registries() -> None:
    reg = generate_unaligned_registries(n_common=60, n_bank_extra=10, n_upi_extra=12, n_telecom_extra=8, seed=3)
    res = align_three_parties(reg.bank_ids, reg.upi_ids, reg.telecom_ids)
    assert res.intersection_size == 60
    for ib, iu, it in zip(res.aligned_indices_bank, res.aligned_indices_upi, res.aligned_indices_telecom):
        assert reg.bank_ids[ib] == reg.upi_ids[iu] == reg.telecom_ids[it]


def test_distance_correlation_properties() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    assert distance_correlation(x, x) == pytest.approx(1.0)
    assert distance_correlation(x, 2.0 * x + 1.0) == pytest.approx(1.0)
    assert distance_correlation(x, rng.normal(size=(200, 2))) < 0.25


@pytest.fixture(scope="module")
def dataset():
    reg = generate_unaligned_registries(n_common=500, seed=7)
    ds, _ = build_aligned_dataset_from_registries(reg, seed=7)
    return ds


def test_split_model_beats_bank_only(dataset) -> None:
    bank, _ = train_bank_only_baseline(dataset.x_bank_train, dataset.y_train, dataset.x_bank_test, dataset.y_test, epochs=60)
    model = SplitCreditNet(defense_config=CutLayerDefenseConfig(enabled=False), seed=7)
    model.fit(dataset.x_bank_train, dataset.x_upi_train, dataset.x_telecom_train, dataset.y_train, epochs=60)
    split = model.evaluate(dataset.x_bank_test, dataset.x_upi_test, dataset.x_telecom_test, dataset.y_test)
    assert split.roc_auc > bank.roc_auc + 0.1


def test_api_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    r = client.post("/psi/align", json={})
    assert r.status_code == 200 and len(r.json()["matched_customer_ids_order"]) == 3
    r = client.post("/score", json={})
    assert r.status_code == 200 and 0.0 <= r.json()["splitnn_default_probability"] <= 1.0
    assert "raw_features_shared_across_wire" not in r.json()["cut_layer_telemetry"]


def test_psi_points_are_in_the_prime_subgroup_and_scalars_are_fresh() -> None:
    import nacl.bindings as sodium

    from splitbureau.psi import ED25519_L, hash_to_group

    for ident in ("PAN_COMMON_00001", "9876543210", "x"):
        h = hash_to_group(ident)
        minus_h = sodium.crypto_scalarmult_ed25519_noclamp((ED25519_L - 1).to_bytes(32, "little"), h)
        assert sodium.crypto_core_ed25519_add(minus_h, h) == bytes([1]) + bytes(31)  # L*H == identity
    # Secret scalars are random per run, so the same IDs give different tokens each time
    r1 = align_two_parties(["a", "b"], ["b", "c"])
    r2 = align_two_parties(["a", "b"], ["b", "c"])
    assert r1.sample_u_hex != r2.sample_u_hex
    assert r1.aligned_indices_a == r2.aligned_indices_a == [1]


def test_defence_reduces_leakage_against_the_stronger_attacker(dataset) -> None:
    from splitbureau.inversion import evaluate_cut_layer_leakage

    def leakage(cfg):
        m = SplitCreditNet(defense_config=cfg, seed=7)
        m.fit(dataset.x_bank_train, dataset.x_upi_train, dataset.x_telecom_train, dataset.y_train, epochs=60)
        return evaluate_cut_layer_leakage(m, dataset.x_bank_train, dataset.x_upi_train, dataset.x_telecom_train,
                                          dataset.x_bank_test, dataset.x_upi_test, dataset.x_telecom_test).overall_mean_r2

    plain = leakage(CutLayerDefenseConfig(enabled=False))
    defended = leakage(CutLayerDefenseConfig(enabled=True))
    # Even without any raw-feature shortcut, an 8-d embedding of 4 features is nearly invertible
    assert plain > 0.7
    assert defended < plain - 0.4


def test_api_key_and_limits(monkeypatch) -> None:
    import splitbureau.api as api

    client = TestClient(app)
    monkeypatch.setattr(api, "API_KEY", "s3cret")
    assert client.post("/score", json={}).status_code == 401
    assert client.get("/health").status_code == 200
    hdr = {"X-API-Key": "s3cret"}
    assert client.post("/score", json={"savings_balance": -1}, headers=hdr).status_code == 422
    big = {"bank_ids": [f"ID{i}" for i in range(api.MAX_PSI_IDS + 1)], "upi_ids": ["a"]}
    assert client.post("/psi/align", json=big, headers=hdr).status_code == 422
    assert client.post("/psi/align", json={"bank_ids": ["x" * 65], "upi_ids": ["a"]}, headers=hdr).status_code == 422
    assert client.post("/train", json={"epochs": 1000}, headers=hdr).status_code == 422
