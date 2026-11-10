"""Download public 2021 contract and injury data with nfl_data_py (nflverse).

nfl_data_py pins pandas<2, so it runs in its own environment:
    python3 -m venv .venv-nfl && .venv-nfl/bin/pip install nfl_data_py
    .venv-nfl/bin/python src/fetch_external.py

Output: outputs/external/contracts_wr.csv, outputs/external/injuries_2021.csv
Sources: nflverse contracts (from OverTheCap) and nflverse injury reports.
"""
from pathlib import Path

import nfl_data_py as nfl

OUT = Path(__file__).resolve().parents[1] / "outputs" / "external"
OUT.mkdir(parents=True, exist_ok=True)

contracts = nfl.import_contracts()
print("contracts:", contracts.shape)
print(list(contracts.columns))
wr = contracts[contracts["position"] == "WR"]
wr.drop(columns=[c for c in ["cols"] if c in wr.columns]).to_csv(OUT / "contracts_wr.csv",
                                                                  index=False)
print("WR contracts:", len(wr))

inj = nfl.import_injuries([2021])
print("injuries:", inj.shape)
print(list(inj.columns))
inj.to_csv(OUT / "injuries_2021.csv", index=False)
