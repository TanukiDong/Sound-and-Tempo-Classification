import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
FILES = {
    "speed.train.joblib": "17goc73izmxmX4m2ouAJY1tjBMR_OtWmS",
    "speed.test.joblib":  "1l5l69jijX6ReH5bHNHdsMHSu52Y2l1KZ",
    "tempo.train.joblib": "1cQwBnTawiF8HxSLSpwo0f0BfYKBvavL9",
    "tempo.test.joblib":  "1SX5W-mI8ok5_8AXpUNAEDKiDLhrKrNJy",
}
TEST_ONLY_FILES = {
    "speed.test.joblib",
    "tempo.test.joblib",
}

def download_files(test: bool) -> None:
    """
    Download dataset files.

    Parameters
    ----------

    test : bool
        If True, download only test datasets.
    """
    # Create the data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    for filename, file_id in FILES.items():

        # If test is True, skip training files
        if test and filename not in TEST_ONLY_FILES:
            continue

        output_dir = DATA_DIR / filename

        # Skip downloading if the file already exists
        if output_dir.exists():
            logger.info("%s already exists, skipping download.", filename)
            continue

        # Download file
        logger.info("Downloading %s .....", filename)
        subprocess.run(
            ["gdown", file_id, "-O", str(output_dir)],
            check=True,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset files.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Download only test data.",
    )
    args = parser.parse_args()

    download_files(args.test)