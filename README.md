[![Run Evaluate Scripts](https://github.com/TanukiDong/Sound-and-Tempo-Classification/actions/workflows/eval.yaml/badge.svg)](https://github.com/TanukiDong/Sound-and-Tempo-Classification/actions/workflows/eval.yaml)

# Sound and Tempo Classification - COM6018 Data Science with Python

A coursework project on developing machine learning systems to detect and classify temporal modifications in speech recordings.

## Overview

This project trains SVM-based classifiers to categorize 1-second speech segments into 5 classes:

- `0`: very slow  
- `1`: slow  
- `2`: normal  
- `3`: fast  
- `4`: very fast  

All code is written in Python 3 and is fully reproducible with the provided scripts.

## Running the Project
```bash
# Clone the repository
git clone https://github.com/TanukiDong/Sound-and-Tempo-Classification.git
cd Sound-and-Tempo-Classification

# Install dependencies
uv sync

# Train speed model
uv run src/train_speed.py

# Evaluate speed model
uv run src/evaluate.py \
    src \
    models/model.speed.joblib \
    data/speed.test.joblib

# Train tempo model
uv run src/train_tempo.py

# Evaluate tempo model
uv run src/evaluate.py \
    src \
    models/model.tempo.joblib \
    data/tempo.test.joblib
```
## Structure

```bash
Sound-and-Tempo-Classification
├── README.md
├── config
│   ├── config.speed.yaml       # Config file for training speed model
│   └── config.tempo.yaml       # Config file for training tempo model
├── images
│   ├── cm_speed.png
│   ├── cm_tempo.png
│   └── data.jpg
├── models
│   ├── model.speed.joblib      # Trained speed model
│   └── model.tempo.joblib      # Trained tempo model
├── pyproject.toml              # Project metadata & dependencies
├── src
│   ├── download_data.py        # Script to download the dataset
│   ├── evaluate.py             # Script to evaluate the trained models
│   ├── train_speed.py          # Script to train the speed model
│   ├── train_tempo.py          # Script to train the tempo model
│   └── utils.py                # Utility functions for data processing and model training
└── uv.lock                     # UV environment lockfile
```

## Dataset

The training and evaluation dataset is extracted from a single female speaker from the CMU Arctic Speech Database ([Kominek and Black, 2004](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.pdf)). 

Each sample is a 1 second utterance that has their speed and tempo modified by :  
- **Speed modification**  
Use `sox speed` command which uniformly changes both pitch and duration.  
- **Tempo modification**  
Use `sox tempo` command which changes duration while preserving pitch.

| Category   | Speed Factor | Tempo Factor |
|------------|--------------|--------------|
| Very Slow  | 0.90         | 0.60         |
| Slow       | 0.95         | 0.80         |
| Normal     | 1.00         | 1.00         |
| Fast       | 1.05         | 1.20         |
| Very Fast  | 1.10         | 1.40         |

The data has been preprocessed from raw signal into filterbank features by the course instructor. The filterbank features represent speech recording as a 2D image, with time on the horizontal axis and frequency on the y axis. They are computed using 64 frequency channels and 10ms time resolution. Therefore, a 1 second speech sample is an image of 64 x 101 pixels.

Example of filterbank feature compared to its original raw signal.

![Data Signal and FBank comparison](https://github.com/TanukiDong/Sound-and-Tempo-Classification/blob/main/images/data.jpg)

The data files can be downloaded using `src/download_data.py`, and the data will be stored in `data/`.

The data is a 4155 x 6464 numpy array, each row represents a flattened filterbank feature matrix of size 64 (frequency channels) x 101 (time frames).

Each training dataset contains 831 speech segments x 5 different categories = 4,155 total samples. Each sample has a label of `[0, 1, 2, 3, 4]`, corresponding to `[very slow, slow, normal, fast, very fast]`.

## Approach

0. **Training Configuration**  
    1. Feature scaler
    2. Data augmentation
    3. Hyperparameter optimisation using HalvingGridSearchCV
    4. Weighting strategy (Voting Classifier)

The training configuration can be modified from the yaml files in `config/`

1. **Speed Model**  
The `train_speed` script trains a single SVM classifier.  
Pipeline :
    1. Frame averaging
    2. Feature scaling
    3. PCA (Optional)
    4. SVM Classifer

2. **Tempo Model**  
The `train_tempo` script trains an ensemble of 5 SVM classifiers, each classifier is trained on different frequency range, with the last model being trained on the full filterbank features.  
Pipeline :
    1. Select frequency band
    2. Average absolute difference (Rate of change)
    3. Feature scaling
    4. PCA (On the full band model)
    5. SVM Classifier
    6. Voting Classifier

Frequency band of each SVM classifier:

| Model | Frequency Channels   |
|-------|----------------------|
| 1     | 0 – 15               |
| 2     | 16 – 31              |
| 3     | 32 – 47              |
| 4     | 48 – 63              |
| 5     | 0 – 63               |

## Experiment & Results

### Effect of data augmentation strategies on validation accuracy, test accuracy, and model size for speed and tempo classification.

Speed Model :
| Augmentation | Validation Accuracy (%) | Test Accuracy (%) | Size (MB) |
|--------------|-------------------------|-------------------|-----------|
| none         | `99.6`                  | `99.8`            | `0.17`    |
| noise        | 99.9                    | 99.6              | 0.20      |
| gain         | 99.9                    | 99.8              | 0.20      |
| noise_gain   | 100.0                   | 99.8              | 0.21      |
| mask         | 82.0                    | 93.4              | 2.19      |

Tempo Model : 
| Augmentation | Validation Accuracy (%) | Test Accuracy (%) | Size (MB) |
|--------------|-------------------------|-------------------|-----------|
| none         | `44.98`                 | `47.40`           | `8.26`    |
| noise        | 42.55                   | 44.20             | 15.82     |
| gain         | 45.65                   | 46.80             | 15.60     |
| noise_gain   | 45.09                   | 47.40             | 15.53     |
| mask         | 44.65                   | 44.00             | 15.93     |

Observations : Data augmentation does not improve the model performance but increase the model size.

### Effect of feature normalisation methods on validation accuracy, test accuracy, and model size for speed and tempo classification

Speed Model :
| Normalisation   | Validation Accuracy (%) | Test Accuracy (%) | Size (MB) |
|-----------------|-------------------------|-------------------|-----------|
| StandardScaler  | `99.6`                  | `99.8`            | `0.17`    |
| MinMaxScaler    | 99.2                    | 99.2              | 0.69      |
| RobustScaler    | 99.6                    | 99.6              | 0.18      |

Tempo Model :
| Normalisation  | Validation Accuracy (%) | Test Accuracy (%) | Size (MB) |
|----------------|-------------------------|-------------------|-----------|
| StandardScaler | `44.98`                 | `47.40`           | `8.26`    |
| MinMaxScaler   | 44.96                   | 46.80             | 8.26      |
| RobustScaler   | 44.97                   | 47.00             | 7.76      |

Observation : StandardScaler gives the best performance for both model.

### Effect of ensemble weighting strategies on validation accuracy, test accuracy, and model size for tempo classification

| Weighting  | Validation Accuracy (%) | Test Accuracy (%) | Size (MB) |
|------------|-------------------------|-------------------|-----------|
| Uniform    | 45.31                   | 45.80             | 7.78      |
| Accuracy   | `44.98`                 | `47.40`           | `8.26`    |
| Full-band  | 44.88                   | 47.20             | 7.81      |
| Ranked     | 44.92                   | 47.56             | 8.23      |

Observation : Accuracy-based weighting gives the best performance.
Note : Further investigation to be done on Accuracy vs Ranked.

### Comparison between baseline and proposed systems in terms of test accuracy and model size for speed and tempo classification.

After multiple model training and evaluation, the best performing model is selected as the proposed system, and it is compared against the baseline KNN model provided by the course instructor.

Speed Model :
| System            | Test Accuracy (%) | Size (MB) |
|-------------------|-------------------|-----------|
| Baseline (KNN)    | 79.2              | 2.11      |
| Proposed System   | 99.8              | 0.17      |


![Speed Confusion Matrix](https://github.com/TanukiDong/Sound-and-Tempo-Classification/blob/main/images/cm_speed.png)

Tempo Model :
| System            | Test Accuracy (%) | Size (MB) |
|-------------------|-------------------|-----------|
| Baseline (KNN)    | 24.6              | 2.11      |
| Proposed System   | 48.2              | 8.23      |

![Tempo Confusion Matrix](https://github.com/TanukiDong/Sound-and-Tempo-Classification/blob/main/images/cm_tempo.png)

## Sources

This project use data from
- CMU Arctic Speech Database [Kominek and Black, 2004](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.pdf)
