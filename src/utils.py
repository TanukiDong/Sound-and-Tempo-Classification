import numpy as np

def avg_abs_frame_diff(X: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Compute per-channel average absolute frame-to-frame differences.

    Parameters
    ----------

    X : np.ndarray, shape (n_samples, n_features) or (n_samples, n_channels * time_frames)  
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