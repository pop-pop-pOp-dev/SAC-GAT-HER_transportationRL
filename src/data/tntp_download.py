import os
import zipfile
from pathlib import Path

import requests


TNTP_BASE_URL = "https://github.com/bstabler/TransportationNetworks/raw/master/"


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dst, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_sioux_falls(data_dir: str) -> dict:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # TNTP structure: dataset folder contains *_net.tntp and *_trips.tntp
    dataset_dir = data_path / "SiouxFalls"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    net_url = TNTP_BASE_URL + "SiouxFalls/SiouxFalls_net.tntp"
    trips_url = TNTP_BASE_URL + "SiouxFalls/SiouxFalls_trips.tntp"

    net_path = dataset_dir / "SiouxFalls_net.tntp"
    trips_path = dataset_dir / "SiouxFalls_trips.tntp"

    download_file(net_url, net_path)
    download_file(trips_url, trips_path)

    return {
        "net_path": str(net_path),
        "trips_path": str(trips_path),
    }


if __name__ == "__main__":
    target_dir = os.environ.get("TNTP_DIR", "./data")
    paths = download_sioux_falls(target_dir)
    print(paths)
