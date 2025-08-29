from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def ensure_path(p: str | Path) -> Path:
    path = Path(p).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def maybe_make_demo_csv(path: str | Path, n: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        rng = np.random.default_rng(42)
        f1, f2, f3 = rng.normal(size=n), rng.uniform(-1, 1, n), rng.normal(1, 0.5, n)
        y = (f1 + 0.8 * f2 - 0.6 * f3 + rng.normal(0, 0.3, n) > 0).astype(int)
        df = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "label": y})
        df.to_csv(path, index=False)
    return path
