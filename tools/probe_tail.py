from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT = "SI VACV"
tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
f_tr = [g.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.smi]
f_te = [g.GetFingerprint(Chem.MolFromSmiles(s)) for s in te.smi]

for i in tr.nlargest(8, TGT).index:
    print("\n" + "=" * 90)
    print(f"TRAIN {tr.Title[i]} | SI={tr[TGT][i]:.0f} | {tr.culture[i]}")
    s_tr = np.array(DataStructs.BulkTanimotoSimilarity(f_tr[i], f_tr))
    nb = [j for j in np.argsort(-s_tr)[1:9] if s_tr[j] >= 0.5]
    print("  соседи в train (sim>=0.5):",
          [(tr.Title[j], round(tr[TGT][j], 1), round(s_tr[j], 2)) for j in nb])
    s_te = np.array(DataStructs.BulkTanimotoSimilarity(f_tr[i], f_te))
    nt = [j for j in np.argsort(-s_te)[:8] if s_te[j] >= 0.5]
    print("  аналоги в test (sim>=0.5):",
          [(int(te['index'][j]), te.Title[j], te.culture[j], round(s_te[j], 2)) for j in nt])