"""
Speed Classification Training Script

This script trains a SVM classifier to classify each 1-second speech segment
into one of five speed categories (very slow, slow, normal, fast, very fast)
from filterbank features.

The training procedure consists of the following steps:
    1. Load training data from a joblib file.
    2. (Optional) Apply data augmentation.
    3. Create a pipeline that performs:
        - Average filterbank frequency channels over time.
        - Apply feature scaling using the selected scaler.
        - (Optional) Apply PCA for dimensionality reduction.
        - Train a SVM classifier.
        - Perform hyperparameter optimisation using 5-fold HalvingGridSearchCV.
    4. Save the trained model as a joblib file.

Usage:

    Command to train the speed model:

        python train_speed.py <TRAINING_DATA_FILE_NAME> <MODEL_FILE_NAME>

        uv run train_speed.py <TRAINING_DATA_FILE_NAME> <MODEL_FILE_NAME> 

    Command to evaluate the trained model:

        python src/evaluate.py <SRC_DIR> <MODEL_FILE_NAME> <TEST_DATA_FILE_NAME>

        uv run src/evaluate.py <SRC_DIR> <MODEL_FILE_NAME> <TEST_DATA_FILE_NAME>

Examples:

To train the model, run:

    Explicit paths:

        python src/train_speed.py \
            data/fbank_speed.train.joblib \
            models/model.speed.joblib

        uv run src/train_speed.py \
            data/fbank_speed.train.joblib \
            models/model.speed.joblib

    Using default paths:

        python src/train_speed.py

        uv run src/train_speed.py


To evaluate the trained model, run:

        python src/evaluate.py \
            src \
            models/model.speed.joblib \
            data/speed.test.joblib

        uv run src/evaluate.py \
            src \
            models/model.speed.joblib \
            data/speed.test.joblib
"""

from pathlib import Path
import numpy as np
import argparse
import joblib
import time
import yaml
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.speed.yaml"
with open(CONFIG_PATH,"r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["seed"]
N_CHANNELS = cfg["features"]["n_channels"]
N_FRAMES = cfg["features"]["n_frames"]
SCALER = cfg["scaler"]
SEARCH_SPACE = cfg["search_space"]
INCLUDE_PCA = cfg["include_pca"]
AUGMENT = cfg["augmentation"]

def augment_data(X: np.ndarray, y: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to the training data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        2D numpy array of flattened filterbank features
    
    y : np.ndarray, shape (n_samples,)
        1D numpy array of target class labels.
    
    mode : str
        Augmentation technique to use.
        One of {"none", "noise", "gain", "noise_gain", "mask"}.
        - none       : No augmentation
        - noise      : Gaussian noise of 10% of standard deviation
        - gain       : Random gain of ±10%
        - noise_gain : Both noise and gain
        - mask       : Time and frequency masking with maximum of 10% of features

    Returns
    -------
    X_aug : np.ndarray, shape (n_samples, n_features) or (2 * n_samples, n_features)
        Augmented filterbank feature matrix.
    
    y_aug : np.ndarray, shape (n_samples,) or (2 * n_samples,)
        Target label of augmented data.

    Raises
    ------
    ValueError
        If "mode" is not a valid augmentation technique.
    """

    mode = mode.lower()

    if mode not in {"none", "noise", "gain", "noise_gain", "mask"}:
        raise ValueError(f"""Invalid augmentation mode: {mode}. \n Choose from "none", "noise", "gain", "noise_gain", "mask".""")
    
    if mode == "none":
        return X, y
    
    # Random Generator
    rng = np.random.default_rng(SEED)

    # Reshape X to (n_samples, N_CHANNELS, N_FRAMES)
    n_samples = X.shape[0]
    X_aug = X.reshape(n_samples, N_CHANNELS, -1).copy()
    
    for i in range(n_samples):
        
        # x = original sample
        x = X_aug[i]

        if mode in {"gain", "noise_gain"}:
            # ± 10% gain
            gain = rng.uniform(0.9, 1.1)
            x = x * gain
        
        if mode in {"noise", "noise_gain"}:
            # 10% of standard deviation
            noise_std = np.std(x) * 0.1
            noise = rng.normal(0, noise_std, x.shape)
            x = x + noise

        if mode == "mask":
            # Time mask
            # 1 ~ 10 = max 10% of 101 frames
            t_range = rng.integers(1, 11)
            t_start = rng.integers(0, x.shape[1] - t_range)
            x[:, t_start:t_start + t_range] = 0

            # Frequency mask
            # 1 ~ 6 = max 10% of 64 channels
            f_range = rng.integers(1, 7)
            f_start = rng.integers(0, x.shape[0] - f_range)
            x[f_start:f_start + f_range, :] = 0

        # Replace with augmented sample
        X_aug[i] = x

    # Flatten back to (n_samples, n_features)
    X_aug = X_aug.reshape(n_samples, -1)

    # Combine original and augmented data
    X_final = np.concatenate([X, X_aug], axis=0)
    y_final = np.concatenate([y, y], axis=0)

    return X_final, y_final

def create_pipeline() -> Pipeline:
    """
    Create a pipeline for speed classification.

    The pipeline applies the following steps in order:
    1. Averaging of filterbank frames
    2. Feature scaling using the selected scaler
    3. (Optional) PCA for dimensionality reduction
    4. SVM classifier

    Returns
    -------
    pipeline : Pipeline
        Configured pipeline.

    See Also
    --------
    average_frames : Function to average filterbank frames.
    select_scaler  : Function to select the scaler.
    """

    scaler = select_scaler(SCALER)

    steps = [
        # 1. Average frames
        ("average_frames", FunctionTransformer(average_frames, validate=False)),
        # 2. Scaler
        ("scaler", scaler),
    ]

    if INCLUDE_PCA:
        # 3. PCA
        steps.append(("pca", PCA(random_state=SEED)))
    
    # 4. SVM classifier
    steps.append(("svm", SVC(random_state=SEED)))

    pipeline = Pipeline(steps)

    return pipeline

def average_frames(X: np.ndarray) -> np.ndarray:
    """
    Reduce the dimensionality of filterbank features to 64-dimensional vector by averaging each features over time.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features) or (n_samples, 64 * time_frames)  
        2D numpy array of flattened filterbank features
        For example, 1 second speech sample with 10ms time resolution (101 time frames) results in (n_samples, 6464)

    Returns
    -------
    np.ndarray, shape (n_samples, 64)
        2D numpy array of averaged frequency of each filterbank features over time.
    """

    # Number of samples
    n_samples = X.shape[0]

    # Return averaged features
    return np.mean(X.reshape(n_samples, N_CHANNELS, -1), axis=2)

def select_scaler(mode: str):
    """
    Select a scaler for the pipeline.

    Parameters
    ----------
    mode : str
        Scaler type to use.
        One of {"standard", "minmax", "robust"}.
        - standard : StandardScaler
        - minmax   : MinMaxScaler
        - robust   : RobustScaler

    Returns
    -------
    scaler : object
        Scaler object corresponding to the selected mode.

    Raises
    ------
    ValueError
        If "mode" is not a valid scaler type.
    """
    if mode not in {"standard", "minmax", "robust"}:
        raise ValueError(f"""Invalid scaler selection : "{mode}". \n Choose from "standard", "minmax", or "robust".""")

    if mode == "standard":
        return StandardScaler()
    if mode == "minmax":
        return MinMaxScaler()
    if mode == "robust":
        return RobustScaler()

def select_param_grid(mode: str):
    """
    Select hyperparameter grid for the pipeline.

        kernel  : Kernel type to be used in the algorithm
        C       : Regularization parameter
        gamma   : Kernel coefficient for rbf

    Parameters
    ----------
    mode : str
        Hyperparameter search space to use.
        One of {"linear", "rbf", "both"}.

    Returns
    -------
    param_grid : list of dict
        Hyperparameter grid corresponding to the selected mode.

    Raises
    ------
    ValueError
        If "mode" is not a valid hyperparameter search space.
    """

    if mode == "linear":
        param_grid = [
            {
                "svm__kernel": ["linear"],
                "svm__C": [0.001, 0.01, 0.1, 1],
            },
        ]
    if mode == "rbf":
        param_grid = [
            {
                "svm__kernel": ["rbf"],
                "svm__C": [0.001, 0.01, 0.1, 1],
                "svm__gamma": ["scale", "auto"],
            },
        ]
    if mode == "both":
        param_grid = [
            # Linear kernel does not require gamma
            {
                "svm__kernel": ["linear"],
                "svm__C": [0.001, 0.01, 0.1, 1],
            },
            {
                "svm__kernel": ["rbf"],
                "svm__C": [0.001, 0.01, 0.1, 1],
                "svm__gamma": ["scale", "auto"],
            },
        ]
    else:
        raise ValueError(f"""Invalid hyperparameter search space selection : "{mode}". \n Choose from "linear", "rbf", or "both".""")
    
    if INCLUDE_PCA:
        # Add PCA n_components to each grid if PCA is included
        for grid in param_grid:
            grid["pca__n_components"] = [16, 24, 32]

    return param_grid

def train(data_file: Path, model_file: Path) -> None:
    """
    Train an SVM model for speed classification by using 5-fold cross validation 
    halving grid search for hyperparameter optimisation.
    
    Optional data augmentation may be applied to the training data.

    Parameters
    ----------
    data_file : pathlib.Path
        Path to speed training data joblib file.
        The file must contain a dictionary with keys:
            "features" : 2D numpy array of shape (n_samples, 64 * time_frames)
                         Each row is a flattened filterbank feature matrix.

            "target"   : 1D numpy array of shape (n_samples,)
                         Each element is the speed class label for the corresponding sample.

    model_file : pathlib.Path
        Path to save the trained speed model as joblib file.
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
    """

    logger.info("Starting training")

    # Create directory if it doesn't exist
    if not model_file.parent.exists():
        model_file.parent.mkdir(parents=True)

    # Load training data
    train_data = joblib.load(data_file)
    # Features
    X = train_data["features"]
    # Target
    y = train_data["target"]
    # Augment data
    X, y = augment_data(X, y, AUGMENT)

    # Pipeline
    pipeline = create_pipeline()

    # Select hyperparameter grid
    param_grid = select_param_grid(SEARCH_SPACE)

    # Initialize HalvingGridSearchCV
    grid_search = HalvingGridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        factor=2, 
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
        random_state=SEED,
    )

    # Train the model
    start_time = time.time()
    grid_search.fit(X, y)
    end_time = time.time()
    

    # Report best hyperparameters
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV accuracy: {grid_search.best_score_}")
    logger.info(f"Training time: {end_time - start_time:.2f} seconds")

    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, model_file)
    logger.info(f"Saved speed model to: {model_file}")

    # Log model size
    model_size = model_file.stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {model_size:.2f} MB")


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Train SVM speed model")

    # Data file
    # Default: data/speed.train.joblib
    parser.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=Path("data/speed.train.joblib"),
        help="Path to training data file (joblib format)",
    )

    # Model output file
    # Default: models/model.speed.joblib
    parser.add_argument(
        "model_file",
        type=Path,
        nargs="?",
        default=Path("models/model.speed.joblib"),
        help="Path to save trained model (joblib format)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Train the model
    train(args.data_file, args.model_file)