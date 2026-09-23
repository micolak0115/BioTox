from typing import List
from rdkit import Chem


possible_atomic_num_list = list(range(1, 119))

possible_chirality_list = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

possible_bonds = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

possible_bond_dirs = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

ATOM_FEATURE_DIM = 2
BOND_FEATURE_DIM = 2
NUM_ATOMIC_NUM = len(possible_atomic_num_list)
NUM_CHIRALITY = len(possible_chirality_list)
NUM_BOND_TYPE = len(possible_bonds)
NUM_BOND_DIR = len(possible_bond_dirs)


def safe_index(choices: List, value) -> int:
    """Return the categorical index, using the last entry for rare unknowns."""
    try:
        return choices.index(value)
    except ValueError:
        return len(choices) - 1


def featurize_atom(atom) -> List[int]:
    return [
        safe_index(possible_atomic_num_list, atom.GetAtomicNum()),
        safe_index(possible_chirality_list, atom.GetChiralTag()),
    ]


def featurize_bond(bond) -> List[int]:
    return [
        safe_index(possible_bonds, bond.GetBondType()),
        safe_index(possible_bond_dirs, bond.GetBondDir()),
    ]
