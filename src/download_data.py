import subprocess
from pathlib import Path

DATA_DIR = Path("data1")
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    "speed.train.joblib": "17goc73izmxmX4m2ouAJY1tjBMR_OtWmS",
    "speed.test.joblib":  "1l5l69jijX6ReH5bHNHdsMHSu52Y2l1KZ",
    "tempo.train.joblib": "1cQwBnTawiF8HxSLSpwo0f0BfYKBvavL9",
    "tempo.test.joblib":  "1SX5W-mI8ok5_8AXpUNAEDKiDLhrKrNJy",
}

for filename, file_id in FILES.items():
    output_dir = DATA_DIR / filename
    if output_dir.exists():
        print(f"{filename} already exists, skipping download.")
        continue

    print(f"Downloading {filename} .....")
    subprocess.run(
        ["gdown", file_id, "-O", str(output_dir)],
        check=True,
    )
