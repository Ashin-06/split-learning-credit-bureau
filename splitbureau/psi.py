"""
Elliptic-Curve Diffie-Hellman Private Set Intersection (DH-PSI), semi-honest model.

Commutative blinding in the prime-order subgroup of Ed25519 (order L ~ 2^252), using
libsodium's point arithmetic through PyNaCl:
  1. Hash-to-group: H(id) = 8 * P, where P is the first SHA-512 candidate encoding that
     decodes to a curve point; multiplying by the cofactor 8 lands in the prime-order subgroup.
  2. Party A blinds its IDs with a fresh random scalar a: u_i = a * H(id_i).
  3. Party B computes v_i = b * u_i and blinds its own IDs: w_j = b * H(id'_j).
  4. Party A computes z_j = a * w_j. v_i == z_j exactly when id_i == id'_j.

What it reveals: the party that compares tokens learns which of its rows are in the
intersection, and both learn its size. In the three-party flow the bank runs two pairwise
PSIs, so it learns its intersection with each other party, not only the three-way one.
This is a reference implementation for study: hand-written protocol code on top of libsodium,
not independently reviewed, and try-and-increment hashing is not constant-time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import secrets
import time
from typing import Dict, List, Sequence

import nacl.bindings as sodium

# Order of the Ed25519 prime-order subgroup
ED25519_L: int = 2**252 + 27742317777372353535851937790883648493


def normalize_identifier(raw_id: str) -> str:
    """Canonicalize a PAN / phone / customer identifier prior to hashing."""
    return raw_id.strip().upper()


def hash_to_group(raw_id: str) -> bytes:
    """Map an identifier to a point of the prime-order subgroup with unknown discrete log."""
    norm = normalize_identifier(raw_id).encode("utf-8")
    counter = 0
    while True:
        candidate = hashlib.sha512(
            b"SPLITBUREAU_DH_PSI_V2:" + norm + b":" + counter.to_bytes(4, "big")
        ).digest()[:32]
        # On-curve and not of small order; may still carry a torsion component
        if sodium.crypto_core_ed25519_is_valid_point(candidate):
            p2 = sodium.crypto_core_ed25519_add(candidate, candidate)
            p4 = sodium.crypto_core_ed25519_add(p2, p2)
            return sodium.crypto_core_ed25519_add(p4, p4)  # 8 * P clears the cofactor
        counter += 1


def _random_scalar() -> bytes:
    """
    A fresh secret scalar in [1, L-1]. Never derived from a seed: anyone who can recompute it
    can hash candidate IDs (phone numbers, PANs) and unblind every token.
    """
    return (1 + secrets.randbelow(ED25519_L - 1)).to_bytes(32, "little")


@dataclasses.dataclass(frozen=True)
class PSIAlignmentResult:
    """Result of two-party Diffie-Hellman Private Set Intersection."""

    party_a_name: str
    party_b_name: str
    party_a_count: int
    party_b_count: int
    intersection_size: int
    jaccard_similarity: float
    aligned_indices_a: List[int]
    aligned_indices_b: List[int]
    aligned_double_blinded_tokens_hex: List[str]
    sample_u_hex: List[str]
    sample_v_hex: List[str]
    sample_w_hex: List[str]
    exact_recovery_rate: float  # checked against the plaintext IDs, possible only in simulation
    elapsed_ms: float


@dataclasses.dataclass(frozen=True)
class ThreePartyPSIResult:
    """Result of 3-party DH-PSI alignment across Bank, UPI App, and Telecom."""

    bank_count: int
    upi_count: int
    telecom_count: int
    intersection_size: int
    aligned_indices_bank: List[int]
    aligned_indices_upi: List[int]
    aligned_indices_telecom: List[int]
    pairwise_bank_upi: PSIAlignmentResult
    pairwise_bank_telecom: PSIAlignmentResult
    exact_recovery_rate: float
    elapsed_ms: float


class DHPSIParty:
    """
    Participant in an elliptic-curve Diffie-Hellman Private Set Intersection protocol.
    Each party holds private identifiers and a fresh secret scalar.
    """

    def __init__(self, name: str, identifiers: Sequence[str]) -> None:
        self.name = name
        self.identifiers: List[str] = [normalize_identifier(x) for x in identifiers]
        self._scalar: bytes = _random_scalar()

    def blind_own_identifiers(self) -> List[bytes]:
        """Compute u_i = k * H(id_i) for each local identifier."""
        return [
            sodium.crypto_scalarmult_ed25519_noclamp(self._scalar, hash_to_group(cid))
            for cid in self.identifiers
        ]

    def exponentiate_peer_tokens(self, peer_tokens: Sequence[bytes]) -> List[bytes]:
        """Multiply incoming blinded tokens by this party's secret scalar."""
        return [sodium.crypto_scalarmult_ed25519_noclamp(self._scalar, tok) for tok in peer_tokens]


def align_two_parties(
    ids_a: Sequence[str],
    ids_b: Sequence[str],
    party_a_name: str = "PartyA_Bank",
    party_b_name: str = "PartyB_UPI",
) -> PSIAlignmentResult:
    """
    Execute the 2-party DH-PSI protocol (both parties simulated in one process) and return
    aligned indices, ordered by Party A's own row order.
    """
    t0 = time.perf_counter()

    party_a = DHPSIParty(party_a_name, ids_a)
    party_b = DHPSIParty(party_b_name, ids_b)

    # Step 1: Party A blinds its own identifiers -> u_i
    u_tokens = party_a.blind_own_identifiers()

    # Step 2: Party B double-blinds u_i -> v_i, and single-blinds its own -> w_j
    v_tokens = party_b.exponentiate_peer_tokens(u_tokens)
    w_tokens = party_b.blind_own_identifiers()

    # Step 3: Party A applies its scalar to Party B's w_j -> z_j
    z_tokens = party_a.exponentiate_peer_tokens(w_tokens)

    # Step 4: Match v_i == z_j
    v_index_map: Dict[bytes, int] = {}
    for idx_a, tok_v in enumerate(v_tokens):
        v_index_map.setdefault(tok_v, idx_a)

    z_index_map: Dict[bytes, int] = {}
    for idx_b, tok_z in enumerate(z_tokens):
        z_index_map.setdefault(tok_z, idx_b)

    # Order by Party A's row index: deterministic, and A's own ordering reveals nothing new
    common_tokens = sorted(set(v_index_map) & set(z_index_map), key=lambda tok: v_index_map[tok])

    aligned_indices_a = [v_index_map[tok] for tok in common_tokens]
    aligned_indices_b = [z_index_map[tok] for tok in common_tokens]

    norm_a = party_a.identifiers
    norm_b = party_b.identifiers
    true_common = set(norm_a) & set(norm_b)

    recovered_pairs_match = all(
        norm_a[ia] == norm_b[ib]
        for ia, ib in zip(aligned_indices_a, aligned_indices_b)
    )
    recovered_set = {norm_a[ia] for ia in aligned_indices_a}
    if len(true_common) == 0:
        exact_recovery = 1.0 if len(recovered_set) == 0 else 0.0
    else:
        exact_recovery = (
            len(recovered_set & true_common) / len(true_common)
            if recovered_pairs_match
            else 0.0
        )

    union_size = len(set(norm_a) | set(norm_b))
    jaccard = len(common_tokens) / union_size if union_size > 0 else 1.0

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return PSIAlignmentResult(
        party_a_name=party_a_name,
        party_b_name=party_b_name,
        party_a_count=len(ids_a),
        party_b_count=len(ids_b),
        intersection_size=len(common_tokens),
        jaccard_similarity=round(jaccard, 6),
        aligned_indices_a=aligned_indices_a,
        aligned_indices_b=aligned_indices_b,
        aligned_double_blinded_tokens_hex=[t.hex()[:24] + "..." for t in common_tokens[:5]],
        sample_u_hex=[t.hex()[:24] + "..." for t in u_tokens[:3]],
        sample_v_hex=[t.hex()[:24] + "..." for t in v_tokens[:3]],
        sample_w_hex=[t.hex()[:24] + "..." for t in w_tokens[:3]],
        exact_recovery_rate=float(exact_recovery),
        elapsed_ms=round(elapsed_ms, 3),
    )


def align_three_parties(
    ids_bank: Sequence[str],
    ids_upi: Sequence[str],
    ids_telecom: Sequence[str],
) -> ThreePartyPSIResult:
    """
    Align common customer records across 3 institutions:
      - Party 1: Retail Bank (coordinator / label holder)
      - Party 2: UPI Super-App
      - Party 3: Telecom & Utility Provider
    The bank runs one pairwise PSI with each other party and keeps the rows in both.
    """
    t0 = time.perf_counter()

    psi_bank_upi = align_two_parties(
        ids_bank,
        ids_upi,
        party_a_name="Party1_Bank",
        party_b_name="Party2_UPI",
    )
    psi_bank_telecom = align_two_parties(
        ids_bank,
        ids_telecom,
        party_a_name="Party1_Bank",
        party_b_name="Party3_Telecom",
    )

    map_bank_to_upi: Dict[int, int] = dict(
        zip(psi_bank_upi.aligned_indices_a, psi_bank_upi.aligned_indices_b)
    )
    map_bank_to_telecom: Dict[int, int] = dict(
        zip(psi_bank_telecom.aligned_indices_a, psi_bank_telecom.aligned_indices_b)
    )

    common_bank_indices = sorted(
        set(map_bank_to_upi.keys()) & set(map_bank_to_telecom.keys())
    )
    aligned_bank = common_bank_indices
    aligned_upi = [map_bank_to_upi[idx] for idx in common_bank_indices]
    aligned_telecom = [map_bank_to_telecom[idx] for idx in common_bank_indices]

    norm_bank = [normalize_identifier(x) for x in ids_bank]
    norm_upi = [normalize_identifier(x) for x in ids_upi]
    norm_telecom = [normalize_identifier(x) for x in ids_telecom]
    true_3way = set(norm_bank) & set(norm_upi) & set(norm_telecom)

    all_equal = all(
        norm_bank[ib] == norm_upi[iu] == norm_telecom[it]
        for ib, iu, it in zip(aligned_bank, aligned_upi, aligned_telecom)
    )
    recovered_ids = {norm_bank[ib] for ib in aligned_bank}
    if len(true_3way) == 0:
        exact_recovery = 1.0 if len(recovered_ids) == 0 else 0.0
    else:
        exact_recovery = (
            len(recovered_ids & true_3way) / len(true_3way) if all_equal else 0.0
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ThreePartyPSIResult(
        bank_count=len(ids_bank),
        upi_count=len(ids_upi),
        telecom_count=len(ids_telecom),
        intersection_size=len(aligned_bank),
        aligned_indices_bank=aligned_bank,
        aligned_indices_upi=aligned_upi,
        aligned_indices_telecom=aligned_telecom,
        pairwise_bank_upi=psi_bank_upi,
        pairwise_bank_telecom=psi_bank_telecom,
        exact_recovery_rate=float(exact_recovery),
        elapsed_ms=round(elapsed_ms, 3),
    )
