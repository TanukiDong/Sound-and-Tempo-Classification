import numpy as np

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

def average_frames(X: np.ndarray, n_channels: int, n_frames: int) -> np.ndarray:
    """
    Reduce the dimensionality of filterbank features to 64-dimensional vector by averaging each features over time.

    Parameters
    ----------

    X : np.ndarray, shape (n_samples, n_features) or (n_samples, n_channels * n_frames)
        2D numpy array of flattened filterbank features
        For example, 1 second speech sample with 10ms time resolution (101 time frames) results in (n_samples, 6464)

    Returns
    -------

    np.ndarray, shape (n_samples, n_channels)
        2D numpy array of averaged frequency of each filterbank features over time.
    """

    # Number of samples
    n_samples = X.shape[0]

    # Return averaged features
    return np.mean(X.reshape(n_samples, n_channels, n_frames), axis=2)

def avg_abs_frame_diff(X: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Compute per-channel average absolute frame-to-frame differences.

    Parameters
    ----------

    X : np.ndarray, shape (n_samples, n_features) or (n_samples, n_channels * n_frames)  
        2D numpy array of flattened filterbank features
    
    Returns
    -------

    np.ndarray, shape (n_samples, n_channels)
        Per-channel change-rate features.
    """
    # Number of samples
    n_samples = X.shape[0]

    # Find differences across time frames for each channel
    diffs = np.diff(X.reshape(n_samples, -1, n_frames), axis=2)

    # Return absolute average difference
    return np.mean(np.abs(diffs), axis=2)


def augment_data(X: np.ndarray, y: np.ndarray, mode: str, n_channels: int, n_frames: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
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
    rng = np.random.default_rng(seed)

    # Reshape X to (n_samples, n_channels, n_frames)
    n_samples = X.shape[0]
    X_aug = X.reshape(n_samples, n_channels, n_frames).copy()
    
    for i in range(n_samples):
        
        # x = original sample
        x = X_aug[i]

        if mode in {"gain", "noise_gain"}:
            # ± 10% gain
            gain = rng.uniform(0.9, 1.1)
            x = x * gain
        
        elif mode in {"noise", "noise_gain"}:
            # 10% of standard deviation
            noise_std = np.std(x) * 0.1
            noise = rng.normal(0, noise_std, x.shape)
            x = x + noise

        else:
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
    elif mode == "minmax":
        return MinMaxScaler()
    else:
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

    else:
        # Emphasize with higher weight and penalize with lower weight according to cv accuracy
        order = np.argsort(scores)
        multiplier = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        scores[order] *= multiplier
        weights = scores / sum(scores)

    return list(weights)