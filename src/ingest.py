import pandas as pd
from ucimlrepo import fetch_ucirepo
from pathlib import Path

def ingest_adult(output_dir: str = 'data/raw') -> dict:
    adult = fetch_ucirepo(id=2)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    adult.data.features.to_parquet(path / 'features.parquet')
    adult.data.targets.to_parquet(path / 'targets.parquet')
    return {'n_rows': len(adult.data.features),
            'n_features': adult.data.features.shape[1],
            'target_dist': adult.data.targets.value_counts().to_dict()}

if __name__ == "__main__":
    result = ingest_adult()
    print(result)
