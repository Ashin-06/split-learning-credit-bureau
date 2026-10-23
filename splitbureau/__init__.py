"""
splitbureau: Vertically Partitioned Private Set Intersection (DH-PSI) &
Split-Learning Credit Bureau with Cut-Layer Feature-Inversion Defense.
"""

from .psi import (
    DHPSIParty,
    PSIAlignmentResult,
    ThreePartyPSIResult,
    align_two_parties,
    align_three_parties,
    SAFE_PRIME_P,
    SUBGROUP_ORDER_Q,
    hash_to_subgroup,
)
from .data import (
    BANK_FEATURES,
    UPI_FEATURES,
    TELECOM_FEATURES,
    VerticalCreditDataset,
    UnalignedBureauRegistries,
    generate_vertical_credit_dataset,
    generate_unaligned_registries,
)
from .splitnn import (
    CutLayerDefenseConfig,
    PartyBottomNet,
    CoordinatorTopNet,
    SplitCreditNet,
    train_bank_only_baseline,
    train_centralized_oracle,
    distance_correlation,
)
from .inversion import (
    FeatureInversionAttack,
    InversionAttackReport,
    evaluate_cut_layer_leakage,
)
from .bench import (
    run_full_benchmark,
    BenchmarkSummary,
)

__version__ = "1.0.0"

__all__ = [
    "DHPSIParty",
    "PSIAlignmentResult",
    "ThreePartyPSIResult",
    "align_two_parties",
    "align_three_parties",
    "SAFE_PRIME_P",
    "SUBGROUP_ORDER_Q",
    "hash_to_subgroup",
    "BANK_FEATURES",
    "UPI_FEATURES",
    "TELECOM_FEATURES",
    "VerticalCreditDataset",
    "UnalignedBureauRegistries",
    "generate_vertical_credit_dataset",
    "generate_unaligned_registries",
    "CutLayerDefenseConfig",
    "PartyBottomNet",
    "CoordinatorTopNet",
    "SplitCreditNet",
    "train_bank_only_baseline",
    "train_centralized_oracle",
    "distance_correlation",
    "FeatureInversionAttack",
    "InversionAttackReport",
    "evaluate_cut_layer_leakage",
    "run_full_benchmark",
    "BenchmarkSummary",
]
