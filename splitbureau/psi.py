"""
Elliptic-Curve / Safe-Prime Diffie-Hellman Private Set Intersection (DH-PSI).

Implements two-party and three-party commutative Diffie-Hellman PSI over the
quadratic-residue subgroup Q_p of an IETF RFC 2409 768-bit safe prime p = 2q + 1:
  1. Hash-to-subgroup: H(id) = (SHA256(id) mod (p - 3) + 2)^2 mod p in Q_p.
  2. Party A blinds its IDs: u_i = H(id_i)^alpha mod p.
  3. Party B double-blinds Party A's tokens: v_i = u_i^beta mod p = H(id_i)^(alpha*beta) mod p,
     and single-blinds its own IDs: w_j = H(id'_j)^beta mod p.
  4. Party A completes the commutative DH on Party B's tokens:
     z_j = w_j^alpha mod p = H(id'_j)^(alpha*beta) mod p.
  5. Both parties align common customers in identical deterministic order without
     revealing any non-intersecting customer IDs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Sequence

# IETF RFC 2409 Group 1 (768-bit Oakley Safe Prime: p = 2q + 1, both p and q prime)
_SAFE_PRIME_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A63A3620FFFFFFFFFFFFFFFF"
)
SAFE_PRIME_P: int = int(_SAFE_PRIME_HEX, 16)
SUBGROUP_ORDER_Q: int = (SAFE_PRIME_P - 1) // 2
# RFC 3526 short-exponent bound (160-bit scalar in Q_p) for fast commutative DH
_SCALAR_BOUND: int = min(SUBGROUP_ORDER_Q - 3, (1 << 160) - 1)


def normalize_identifier(raw_id: str) -> str:
    """Canonicalize a PAN / phone / customer identifier prior to hashing."""
    return raw_id.strip().upper()


def hash_to_subgroup(raw_id: str, p: int = SAFE_PRIME_P) -> int:
    """
    Map a string identifier deterministically into the prime-order quadratic
    residue subgroup Q_p of order q = (p - 1) / 2.

    Squaring modulo a safe prime p = 2q + 1 guarantees Legendre symbol +1,
    preventing quadratic-residiosity leakage.
    """
    norm = normalize_identifier(raw_id).encode("utf-8")
    digest = hashlib.sha256(b"SPLITBUREAU_DH_PSI_V1:" + norm).digest()
    h_int = int.from_bytes(digest, byteorder="big")
    base = (h_int % (p - 3)) + 2
    return pow(base, 2, p)


def _derive_secret_exponent(
    q: int = SUBGROUP_ORDER_Q,
    seed: Optional[int] = None,
    party_tag: str = "party",
) -> int:
    """Generate a private exponent in [2, min(q - 2, 2^160)]."""
    bound = min(q - 3, _SCALAR_BOUND)
    if seed is None:
        return secrets.randbelow(bound) + 2
    digest = hashlib.sha256(f"DH_PSI_KEY:{party_tag}:{seed}".encode("utf-8")).digest()
    val = int.from_bytes(digest, byteorder="big")
    return (val % bound) + 2


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
    exact_recovery_rate: float
    non_overlap_leaked: int
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
    non_overlap_leaked: int
    elapsed_ms: float


class DHPSIParty:
    """
    Participant in a Safe-Prime Diffie-Hellman Private Set Intersection protocol.
    Each party holds private identifiers and a secret exponent in Z_q*.
    """

    def __init__(
        self,
        name: str,
        identifiers: Sequence[str],
        seed: Optional[int] = None,
        p: int = SAFE_PRIME_P,
        q: int = SUBGROUP_ORDER_Q,
    ) -> None:
        self.name = name
        self.identifiers: List[str] = [normalize_identifier(x) for x in identifiers]
        self.p = p
        self.q = q
        self._exponent: int = _derive_secret_exponent(q=q, seed=seed, party_tag=name)

    def blind_own_identifiers(self) -> List[int]:
        """Compute u_i = H(id_i)^alpha mod p for each local identifier."""
        p = self.p
        exp = self._exponent
        return [pow(hash_to_subgroup(cid, p), exp, p) for cid in self.identifiers]

    def exponentiate_peer_tokens(self, peer_tokens: Sequence[int]) -> List[int]:
        """Raise incoming blinded tokens to this party's secret exponent mod p."""
        p = self.p
        exp = self._exponent
        return [pow(tok, exp, p) for tok in peer_tokens]


def align_two_parties(
    ids_a: Sequence[str],
    ids_b: Sequence[str],
    party_a_name: str = "PartyA_Bank",
    party_b_name: str = "PartyB_UPI",
    seed: Optional[int] = 42,
) -> PSIAlignmentResult:
    """
    Execute the 2-party Safe-Prime DH-PSI protocol and return aligned indices
    in identical order without revealing non-intersecting IDs.
    """
    t0 = time.perf_counter()

    seed_a = None if seed is None else seed * 101 + 7
    seed_b = None if seed is None else seed * 103 + 13

    party_a = DHPSIParty(party_a_name, ids_a, seed=seed_a)
    party_b = DHPSIParty(party_b_name, ids_b, seed=seed_b)

    # Step 1: Party A blinds its own identifiers -> u_i
    u_tokens = party_a.blind_own_identifiers()

    # Step 2: Party B double-blinds u_i -> v_i, and single-blinds its own -> w_j
    v_tokens = party_b.exponentiate_peer_tokens(u_tokens)
    w_tokens = party_b.blind_own_identifiers()

    # Step 3: Party A raises Party B's w_j to alpha -> z_j
    z_tokens = party_a.exponentiate_peer_tokens(w_tokens)

    # Step 4: Match v_i == z_j and align in deterministic canonical order
    v_index_map: Dict[int, int] = {}
    for idx_a, tok_v in enumerate(v_tokens):
        if tok_v not in v_index_map:
            v_index_map[tok_v] = idx_a

    z_index_map: Dict[int, int] = {}
    for idx_b, tok_z in enumerate(z_tokens):
        if tok_z not in z_index_map:
            z_index_map[tok_z] = idx_b

    common_tokens = sorted(set(v_index_map.keys()) & set(z_index_map.keys()))

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

    non_overlap_z = set(z_index_map.keys()) - set(v_index_map.keys())
    non_overlap_leaked = len(non_overlap_z & set(v_index_map.keys()))

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
        aligned_double_blinded_tokens_hex=[hex(t)[:26] + "..." for t in common_tokens[:5]],
        sample_u_hex=[hex(t)[:26] + "..." for t in u_tokens[:3]],
        sample_v_hex=[hex(t)[:26] + "..." for t in v_tokens[:3]],
        sample_w_hex=[hex(t)[:26] + "..." for t in w_tokens[:3]],
        exact_recovery_rate=float(exact_recovery),
        non_overlap_leaked=int(non_overlap_leaked),
        elapsed_ms=round(elapsed_ms, 3),
    )


def align_three_parties(
    ids_bank: Sequence[str],
    ids_upi: Sequence[str],
    ids_telecom: Sequence[str],
    seed: int = 42,
) -> ThreePartyPSIResult:
    """
    Align common customer records across 3 institutions:
      - Party 1: Retail Bank (coordinator / label holder)
      - Party 2: UPI Super-App
      - Party 3: Telecom & Utility Provider
    """
    t0 = time.perf_counter()

    psi_bank_upi = align_two_parties(
        ids_bank,
        ids_upi,
        party_a_name="Party1_Bank",
        party_b_name="Party2_UPI",
        seed=seed,
    )
    psi_bank_telecom = align_two_parties(
        ids_bank,
        ids_telecom,
        party_a_name="Party1_Bank",
        party_b_name="Party3_Telecom",
        seed=seed + 100,
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

    non_overlap_leaked = (
        psi_bank_upi.non_overlap_leaked + psi_bank_telecom.non_overlap_leaked
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
        non_overlap_leaked=int(non_overlap_leaked),
        elapsed_ms=round(elapsed_ms, 3),
    )
