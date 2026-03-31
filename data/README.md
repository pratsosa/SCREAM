# Data

Real data lives in `$PSCRATCH` (`/pscratch/sd/p/pratsosa/`) and is never committed to this repository.

| File | Description |
|---|---|
| `$PSCRATCH/GD1_errs_LS_CATHODE_withsf_v2.csv` | Raw GD-1 catalog (input to flow training) |
| `$PSCRATCH/gd1_cathode/gd1_generated.csv` | Generated background samples from pzflow (output of `scripts/train_flow.py`) |

See `configs/streams/gd1.yaml` for the authoritative paths used by training scripts.
