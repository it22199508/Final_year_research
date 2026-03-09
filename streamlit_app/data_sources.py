from pathlib import Path
from typing import List

def list_csv_instances() -> List[Path]:
    paths=[]
    for d in [Path("data/processed"), Path("data/test")]:
        if d.exists():
            paths += sorted(d.glob("*.csv"))
    # de-dup
    seen=set(); out=[]
    for p in paths:
        s=p.as_posix()
        if s not in seen:
            out.append(p); seen.add(s)
    return out
