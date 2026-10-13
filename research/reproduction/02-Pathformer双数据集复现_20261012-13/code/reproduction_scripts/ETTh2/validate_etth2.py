import hashlib
from pathlib import Path

import pandas as pd

path = Path("/root/autodl-tmp/pathformer_repro/pathformer/dataset/ETT/ETTh2.csv")
df = pd.read_csv(path)
expected_columns = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
assert list(df.columns) == expected_columns, df.columns.tolist()
assert df.shape == (17420, 8), df.shape
assert int(df.isna().sum().sum()) == 0
timestamps = pd.to_datetime(df["date"])
assert bool((timestamps.diff().iloc[1:] == pd.Timedelta(hours=1)).all())
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"shape={df.shape}")
print(f"range={timestamps.iloc[0]} to {timestamps.iloc[-1]}")
print(f"missing_values={int(df.isna().sum().sum())}")
print(f"sha256={digest}")
