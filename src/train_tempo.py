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

Usage:

    Command to train the tempo model:

        python train_tempo.py <TRAINING_DATA_FILE_NAME> <MODEL_FILE_NAME>

        uv run train_tempo.py <TRAINING_DATA_FILE_NAME> <MODEL_FILE_NAME>

    Command to evaluate the trained model:

        python src/evaluate.py <SRC_DIR> <MODEL_FILE_NAME> <TEST_DATA_FILE_NAME>

        uv run src/evaluate.py <SRC_DIR> <MODEL_FILE_NAME> <TEST_DATA_FILE_NAME> 

Examples:

To train the model, run:

    Explicit paths:

        python src/mymodel_fbank/train_tempo.py \
            data/fbank_tempo.train.joblib \
            models/mymodel_fbank/model.tempo.joblib

        uv run src/mymodel_fbank/train_tempo.py \
            data/fbank_tempo.train.joblib \
            models/mymodel_fbank/model.tempo.joblib

    Using default paths:

        python src/mymodel_fbank/train_tempo.py

        uv run src/mymodel_fbank/train_tempo.py


To evaluate the trained model, run:

        python src/evaluate.py \
            src/mymodel_fbank \
            models/mymodel_fbank/model.tempo.joblib \
            data/fbank_tempo.test1.joblib 

        uv run src/evaluate.py \
            src/mymodel_fbank \
            models/mymodel_fbank/model.tempo.joblib \
            data/fbank_tempo.test1.joblib
"""

from pathlib import Path
import numpy as np
import argparse
import joblib
import time

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.ensemble import VotingClassifier

# Random seed
SEED = 42

# Frequency channels of filterbank features
N_CHANNELS = 64
# Number of frames per sample
N_FRAMES = 101

# Select Scaler
# 1. standard : StandardScaler
# 2. minmax   : MinMaxScaler
# 3. robust   : RobustScaler
SCALER = "standard"

# Select hyperparameter search space
# 1. linear   : Linear kernel only
# 2. rbf      : RBF kernel only
# 3. both     : Both linear and RBF kernels
SEARCH_SPACE = "both"

# Select data augmentation technique
# 1. none       : No data augmentation
# 2. noise      : Noise only
# 3. gain       : Gain only
# 4. noise_gain : Both noise and gain
# 5. mask       : Masking only
AUGMENT = "none"

# Select weighting strategy for the VotingClassifier
# 1. equal      : Equal weights
# 2. accuracy   : Weights based on CV accuracy
# 3. fullband   : Full band gets higher weight
# 4. rank       : Emphasize/penalize according to rank
WEIGHT = "accuracy"

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

def select_weight(scores, mode: str) -> list[float]:
    """
    Calculate ensemble weights for the VotingClassifier.

    Parameters
    ----------
    scores : np.ndarray, shape (n_classifiers,)
        Cross-validation accuracy scores of each classifier.

    mode : str
        Weighting strategy to use.
        One of {"equal", "accuracy", "fullband", "rank"}.
        - equal      : Equal weights
        - accuracy   : Weights based on CV accuracy
        - fullband   : Full band gets higher weight
        - rank       : Emphasize/penalize according to rank

    Returns
    -------
    weights : List[float]
        Weight list corresponding to the selected strategy.

    Raises
    ------
    ValueError
        If "mode" is not a valid weighting strategy.
    """
    if mode not in {"equal", "accuracy", "fullband", "rank"}:
        raise ValueError(f"""Invalid weighting strategy : "{mode}". \n Choose from "equal", "accuracy", "fullband", or "rank".""")

    if mode == "equal":
        # Equal weights
        weights = np.ones_like(scores) / len(scores)

    elif mode == "accuracy":
        # Weights based on CV accuracy
        weights = scores / sum(scores)

    elif mode == "fullband":
        # Give the full band 1.5x weight
        scores[-1] *= 1.5
        weights = scores / sum(scores)

    elif mode == "rank":
        # Emphasize with higher weight and penalize with lower weight according to cv accuracy
        order = np.argsort(scores)
        multiplier = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        scores[order] *= multiplier
        weights = scores / sum(scores)

    return list(weights)

def create_pipeline(start_ch: int, end_ch: int) -> Pipeline:
    """
    Create a pipeline for tempo classification.

    The pipeline consists of:
    1. Select columns for the specified band
    2. Feature scaling using the selected scaler
    3. PCA for dimensionality reduction
    4. SVM classifier

    Parameters
    ----------
    start_ch : int
        Starting channel index (inclusive) for the band.

    end_ch : int
        Ending channel index (exclusive) for the band.

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

    # Keep only selected band columns
    # And drop the rest
    select_band = ColumnTransformer(
        transformers=[("band", "passthrough", band_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Select scaler for the pipeline
    scaler = select_scaler(SCALER)

    # Pipeline
    pipeline = Pipeline(
        [
            # 1. Select columns for the band
            ("band", select_band),
            # 2. Scaler
            ("scaler", scaler),
            # 3. PCA
            ("pca", PCA(random_state=SEED)),
            # 4. SVM classifier
            ("svm", SVC(
                probability=True,
                random_state=SEED,
                )
            ),
            
        ]
    )

    return pipeline

def select_param_grid(mode: str):
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
        return [
            {
                "pca__n_components": [16, 24, 32],
                "svm__kernel": ["linear"],
                "svm__C": [0.001, 0.01, 0.1, 1],
            },
        ]
    if mode == "rbf":
        return [
            {
                "pca__n_components": [16, 24, 32],
                "svm__kernel": ["rbf"],
                "svm__C": [0.001, 0.01, 0.1, 1],
                "svm__gamma": ["scale", "auto"],
            },
        ]
    if mode == "both":
        return [
            # Linear kernel does not require gamma
            {
                "pca__n_components": [16, 24, 32],
                "svm__kernel": ["linear"],
                "svm__C": [0.001, 0.01, 0.1, 1],
            },
            {
                "pca__n_components": [16, 24, 32],
                "svm__kernel": ["rbf"],
                "svm__C": [0.001, 0.01, 0.1, 1],
                "svm__gamma": ["scale", "auto"],
            },
        ]
    else:
        raise ValueError(f"""Invalid hyperparameter search space selection : "{mode}". \n Choose from "linear", "rbf", or "both".""")
    
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
            "features" : 2D numpy array of shape (n_samples, 64 * time_frames)
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

    # Debug message
    print("\033[94mStarting training\033[0m")

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
    X, y = augment_data(X, y, AUGMENT)

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

    param_grid = select_param_grid(SEARCH_SPACE)

    estimators = []
    scores = []
    
    # Train one model per band, total 5 models
    time_start = time.time()
    for idx, (start_ch, end_ch) in enumerate(bands, start=1):
        print(f"Model {idx}/5 : channels {start_ch}-{end_ch - 1}")

        # Create pipeline
        pipeline = create_pipeline(start_ch, end_ch)

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
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_}")
        print(f"Training time: {time_end_1 - time_start_1:.2f} seconds")

    # Convert to numpy array
    scores = np.array(scores, dtype=float)

    # Select weights for the VotingClassifier
    weights = select_weight(scores, WEIGHT)

    # Debug message
    print(f"Normalised weights for VotingClassifier: {weights}")

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

    print("\033[94mTraining soft voter\033[0m")
    ensemble_start = time.time()
    voter.fit(X, y)
    ensemble_end = time.time()

    print(f"Voter training time: {ensemble_end - ensemble_start:.2f} seconds")

    time_end = time.time()
    print(f"Total training time: {time_end - time_start:.2f} seconds")

    joblib.dump(voter, model_file)
    print(f"Saved tempo model to: {model_file}")

    # Print model size
    model_size = model_file.stat().st_size / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="Train SVM tempo model")

    # Data file
    # Default: data/fbank_tempo.train.joblib
    parser.add_argument(
        "data_file",
        type=Path,
        nargs="?",
        default=Path("data/fbank_tempo.train.joblib"),
        help="Path to tempo training data file (joblib format)",
    )

    # Model output file
    # Default: models/mymodel_fbank/model.tempo.joblib
    parser.add_argument(
        "model_file",
        type=Path,
        nargs="?",
        default=Path("models/mymodel_fbank/model.tempo.joblib"),
        help="Path to save trained tempo model (joblib format)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Train the model
    train(args.data_file, args.model_file)