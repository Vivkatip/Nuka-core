"""Стандартизация SMILES и признаки RDKit. CC50/IC50 не используются."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

FP_BITS = 2048
MACCS_BITS = 167
CULTURE_LEVELS = ["Vero", "HFF", "HeLa", "A549", "HEL", "BSC", "CEM",
                  "MT-4", "E6SM", "Other", "Unknown"]

_UNCHARGER = rdMolStandardize.Uncharger()
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
_DESCS = [(n, f) for n, f in Descriptors.descList if n != "Ipc"]
DESC_NAMES = [n for n, _ in _DESCS]


def standardize(smiles):
    """Родительский нейтральный органический фрагмент или None."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    text = smiles.strip()
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None and "N(=O)=O" in text:
            mol = Chem.MolFromSmiles(text.replace("N(=O)=O", "[N+](=O)[O-]"))
        if mol is None:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        organic = [f for f in frags if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
        parent = max(organic or list(frags), key=lambda m: m.GetNumHeavyAtoms())
        try:
            parent = _UNCHARGER.uncharge(parent)
        except Exception:
            pass
        return parent
    except Exception:
        return None


def canonical_key(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def _descriptors(mol):
    out = np.full(len(_DESCS), np.nan)
    for i, (_, fn) in enumerate(_DESCS):
        try:
            v = float(fn(mol))
            if np.isfinite(v) and abs(v) < 1e8:
                out[i] = v
        except Exception:
            pass
    return out


def culture_onehot(names):
    idx = {c: i for i, c in enumerate(CULTURE_LEVELS)}
    out = np.zeros((len(names), len(CULTURE_LEVELS)))
    for r, name in enumerate(names):
        out[r, idx.get(name, idx["Other"])] = 1.0
    return out


def featurize(mols):
    """mols: список RDKit-молекул или None. Возвращает словарь матриц."""
    n = len(mols)
    desc = np.full((n, len(_DESCS)), np.nan)
    fp = np.zeros((n, FP_BITS), dtype=np.uint8)
    maccs = np.zeros((n, MACCS_BITS), dtype=np.uint8)
    valid = np.zeros(n, dtype=bool)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        valid[i] = True
        desc[i] = _descriptors(mol)
        fp[i] = _MORGAN.GetFingerprintAsNumPy(mol)
        arr = np.zeros(MACCS_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
        maccs[i] = arr
    return {"desc": desc, "fp": fp, "maccs": maccs, "valid": valid}
