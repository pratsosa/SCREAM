import os
from datetime import datetime, timezone
from pathlib import Path


def date_string() -> str:
    """Return a compact UTC timestamp string suitable for run/file naming.

    Example: '20260331_143022'
    """
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def get_scratch_dir(stream_name: str) -> Path:
    """Return the root scratch directory for a given stream.

    All SCREAM outputs for a stream live under $PSCRATCH/<STREAM>_SCREAM/.
    Subdirectories:
        generated/    — flow-generated background samples (CSV)
        checkpoints/  — model checkpoints (.ckpt)
        loaders/      — serialized DataLoaders (.pth)
        results/      — prediction CSVs and evaluation outputs

    Parameters
    ----------
    stream_name : str
        Stream identifier (e.g. "gd1"). Case-insensitive; uppercased internally.

    Returns
    -------
    Path
        $PSCRATCH/<STREAM>_SCREAM  (not guaranteed to exist yet — callers must
        create subdirectories as needed with mkdir(parents=True, exist_ok=True))
    """
    pscratch = os.environ.get("PSCRATCH")
    if pscratch is None:
        raise EnvironmentError(
            "PSCRATCH environment variable is not set. "
            "Are you running on a NERSC system with the environment loaded?"
        )
    return Path(pscratch) / f"{stream_name.upper()}_SCREAM"
