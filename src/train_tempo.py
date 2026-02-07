"""
Tempo Classification Training Script

This script trains an ensemble model of 5 SVM classifiers to classify each 1-second
speech segment into one of five tempo categories (very slow, slow, normal, fast, very fast)
from filterbank features.

The ensemble consists of 4 SVM classifiers, each trained on a non-overlapping 16 channel frequency band
of the filterbank features, and 1 SVM classifier trained on the full 64 channel frequency band.
They are combined using a soft voting classifier.

The training procedure consists of the following steps:
    1. Load training data from a joblib file.
    2. (Optional) Apply data augmentation.
    3. Train 5 SVM classifiers, each on a different frequency band:
        - Select frequency range.
        - Apply feature scaling using the selected scaler.
        - Apply PCA for dimensionality reduction.
        - Train a SVM classifier.
        - Perform hyperparameter optimisation using 5-fold HalvingGridSearchCV.
        - Save the best estimator and its CV score.
    4. Calculate ensemble weights for the classifiers.
    5. Combine the 5 optimised classifiers using a soft voting classifier.
    6. Save the trained ensemble model as a joblib file.

Usage
-----

    Command to train the tempo model:

        python train_tempo.py <TRAINING_DATA_FILE_NAME> <MODEL_FILE_NAME>

        uv run train_tempo.py <TRAINING_DATA_FILE_NAME> <MODEL_FILE_NAME>

    Command to evaluate the trained model:

        python src/evaluate.py <SRC_DIR> <MODEL_FILE_NAME> <TEST_DATA_FILE_NAME>

        uv run src/evaluate.py <SRC_DIR> <MODEL_FILE_NAME> <TEST_DATA_FILE_NAME> 

Examples
--------

To train the model, run:

    Explicit paths:

        python src/train_tempo.py \
            data/tempo.train.joblib \
            models/model.tempo.joblib

        uv run src/train_tempo.py \
            data/tempo.train.joblib \
            models/model.tempo.joblib

    Using default paths:

        python src/train_tempo.py

        uv run src/train_tempo.py


To evaluate the trained model, run:

        python src/evaluate.py \
            src \
            models/model.tempo.joblib \
            data/tempo.test.joblib 

        uv run src/evaluate.py \
            src \
            models/model.tempo.joblib \
            data/tempo.test.joblib
"""

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from utils import (augment_data, avg_abs_frame_diff, select_scaler,
                   select_weight)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Load configuration parameters
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.tempo.yaml"
with open(CONFIG_PATH,"r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["seed"]
N_CHANNELS = cfg["features"]["n_channels"]
N_FRAMES = cfg["features"]["n_frames"]
SCALER = cfg["scaler"]
SEARCH_SPACE = cfg["search_space"]
AUGMENT = cfg["augmentation"]
WEIGHT = cfg["voting_weight"]

def create_pipeline(start_ch: int, end_ch: int, include_pca: bool) -> Pipeline:
    """
    Create a pipeline for tempo classification.

    The pipeline consists of:
    1. Select columns for the specified band
    2. Calculate rate of change using average absolute frame-to-frame differences
    3. Feature scaling using the selected scaler
    4. PCA for dimensionality reduction (for full band model)
    5. SVM classifier

    Parameters
    ----------

    start_ch : int
        Starting channel index (inclusive) for the band.

    end_ch : int
        Ending channel index (exclusive) for the band.

    include_pca : bool
        Whether to include PCA in the pipeline.
        Use for full band model.

    Returns
    -------

    pipeline : Pipeline
        Pipeline for tempo classification.

    See Also
    --------

    select_scaler : Function to select the scaler.
    """

    # Column indices for the selected band
    band_cols = np.arange(start_ch * N_FRAMES, end_ch * N_FRAMES)

    # Keep only selected band columns and drop the rest
    select_band = ColumnTransformer(
        transformers=[("band", "passthrough", band_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Select scaler for the pipeline
    scaler = select_scaler(SCALER)

    steps = [
            # 1. Select columns for the band
            ("band", select_band),
            # 2. Calculate change-rate features
            ("change_rate", FunctionTransformer(avg_abs_frame_diff, kw_args={"n_frames": N_FRAMES}, validate=False)),
            # 3. Scaler
            ("scaler", scaler),
        ]

    # 4. PCA on the full band model
    if include_pca:
        steps.append(("pca", PCA(random_state=SEED)))

    # 5. SVM classifier
    steps.append(("svm",SVC(probability=True,random_state=SEED)))
    
    return Pipeline(steps)

def select_param_grid(mode: str, include_pca: bool) -> list[dict]:
    """
    Select hyperparameter grid for the pipeline.

        PCA parameters
            n_components : Number of PCA components to keep
        SVM parameters
            kernel  : Kernel type to be used in the algorithm
            C       : Regularization parameter
            gamma   : Kernel coefficient for rbf

    Parameters
    ----------

    mode : str
        Hyperparameter search space to use.
        One of {"linear", "rbf", "both"}.

    include_pca : bool
        Whether to include PCA in the pipeline.
        Use for full band model.

    Returns
    -------

    param_grid : list of dict
        Hyperparameter grid corresponding to the selected mode.

    Raises
    ------

    ValueError
        If "mode" is not a valid hyperparameter search space.
    """
    if mode not in {"linear", "rbf", "both"}:
        raise ValueError(f"""Invalid hyperparameter search space : "{mode}". \n Choose from "linear", "rbf", or "both".""")

    if mode == "linear":
        param_grid = [
            {
                # "pca__n_components": [16, 24, 32],
                "svm__kernel": ["linear"],
                "svm__C": [0.001, 0.01, 0.1, 1],
            },
        ]
    elif mode == "rbf":
        param_grid = [
            {
                # "pca__n_components": [16, 24, 32],
                "svm__kernel": ["rbf"],
                "svm__C": [0.001, 0.01, 0.1, 1],
                "svm__gamma": ["scale", "auto"],
            },
        ]
    else:
        param_grid = [
            # Linear kernel does not require gamma
            {
                # "pca__n_components": [16, 24, 32],
                "svm__kernel": ["linear"],
                "svm__C": [0.001, 0.01, 0.1, 1],
            },
            {
                # "pca__n_components": [16, 24, 32],
                "svm__kernel": ["rbf"],
                "svm__C": [0.001, 0.01, 0.1, 1],
                "svm__gamma": ["scale", "auto"],
            },
        ]

    if include_pca:
        # Add PCA n_components to parameter grids
        for grid in param_grid:
            grid["pca__n_components"] = [8, 16, 24]

    return param_grid
    
def train(data_file: Path, model_file: Path) -> None:
    """
    Train an ensemble model of 5 SVM classifiers of varying frequency bands 
    for tempo classification by using 5-fold cross validation halving grid 
    search for hyperparameter optimisation and soft voting for final prediction.

    5 frequency bands:
        1. Channels 0 - 15
        2. Channels 16 - 31
        3. Channels 32 - 47
        4. Channels 48 - 63
        5. Channels 0 - 63 (full band)

    Optional data augmentation may be applied to the training data.

    Parameters
    ----------

    data_file : pathlib.Path
        Path to the tempo training data joblib file.
        The file must contain a dictionary with keys:
            "features" : 2D numpy array of shape (n_samples, n_channels * time_frames)
                         Each row is a flattened filterbank feature matrix.

            "target"   : 1D numpy array of shape (n_samples,)
                         Each element is the tempo class label for the corresponding sample.

    model_file : pathlib.Path
        Path to save the trained tempo model as joblib file.
        The directory will be created automatically if it does not exist.

    Returns
    -------

    None
        The trained model is saved to model_file.

    See Also
    --------

    augment_data      : Function to apply data augmentation to training data.
    select_param_grid : Function to select hyperparameter search space.
    create_pipeline   : Function to create the SVM pipeline.
    select_weight     : Function to calculate weights for VotingClassifier.
    """

    logger.info("Starting training")

    # Create directory if it doesn't exist
    if not model_file.parent.exists():
        model_file.parent.mkdir(parents=True)

    # Load training data
    train_data = joblib.load(data_file)
    # Features
    X = train_data.get("features")
    # Target
    y = train_data.get("target")
    # Augment data
    X, y = augment_data(X, y, AUGMENT, N_CHANNELS, N_FRAMES, SEED)

    # Different bands for each model
    # 4 sub-bands of 16 channels
    # 1 full band of 64 channels
    bands = [
        (0, 16),
        (16, 32),
        (32, 48),
        (48, 64),
        (0, 64),
    ]

    estimators = []
    scores = []
    include_pca = False
    
    # Train one model per band, total 5 models
    time_start = time.time()
    for idx, (start_ch, end_ch) in enumerate(bands, start=1):

        logger.info("Training model %d/5 | channels %d-%d", idx, start_ch, end_ch - 1)

        # Include PCA only for the full band model
        if idx == 5:
            include_pca = True

        # Create pipeline
        pipeline = create_pipeline(start_ch, end_ch, include_pca=include_pca)
        param_grid = select_param_grid(SEARCH_SPACE, include_pca=include_pca)

        # Initialize HalvingGridSearchCV
        grid_search = HalvingGridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            factor=2,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )

        # Train the model
        time_start_1 = time.time()
        grid_search.fit(X, y)
        time_end_1 = time.time()

        # Store best estimator and score
        estimators.append(grid_search.best_estimator_)
        scores.append(grid_search.best_score_)

        # Report best hyperparameters
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV accuracy: {grid_search.best_score_}")
        logger.info(f"Training time: {time_end_1 - time_start_1:.2f} seconds")

    # Convert to numpy array
    scores = np.array(scores, dtype=float)

    # Select weights for the VotingClassifier
    weights = select_weight(scores, WEIGHT)

    # Debug message
    logger.info(f"Normalised weights for VotingClassifier: {weights}")

    # Prepare models for the VotingClassifier
    models = [
        (f"model {i+1}", estimator) for i, estimator in enumerate(estimators)
    ]

    # Soft VotingClassifier with weights
    voter = VotingClassifier(
        estimators=models,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )

    logger.info("Training soft voter")
    ensemble_start = time.time()
    voter.fit(X, y)
    ensemble_end = time.time()

    logger.info(f"Voter training time: {ensemble_end - ensemble_start:.2f} seconds")

    time_end = time.time()
    logger.info(f"Total training time: {time_end - time_start:.2f} seconds")

    joblib.dump(voter, model_file)
    logger.info(f"Saved tempo model to: {model_file}")

    # Log model size
    model_size = model_file.stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {model_size:.2f} MB")


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Train SVM tempo model")

    # Data file
    # Default: data/tempo.train.joblib
    parser.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=Path("data/tempo.train.joblib"),
        help="Path to tempo training data file (joblib format)",
    )

    # Model output file
    # Default: models/model.tempo.joblib
    parser.add_argument(
        "model_file",
        type=Path,
        nargs="?",
        default=Path("models/model.tempo.joblib"),
        help="Path to save trained tempo model (joblib format)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Train the model
    train(args.data_file, args.model_file)
