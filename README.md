[![Run Evaluate Scripts](https://github.com/TanukiDong/Sound-and-Tempo-Classification/actions/workflows/run-scripts.yaml/badge.svg)](https://github.com/TanukiDong/Sound-and-Tempo-Classification/actions/workflows/run-scripts.yaml)

# Sound and Tempo Classification - COM6018 Data Science with Python

A coursework project on developing machine learning systems for detecting and classifying temporal modifications in speech recordings.

## Overview

This project implements SVM classifier as the main machine learning algorithm to classify speech recording into 5 categories (very slow, slow, normal, fast, very fast).

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
│   └── config.tempo.yaml
├── images
│   └── data.jpg
├── models
│   ├── model.speed.joblib
│   └── model.tempo.joblib
├── pyproject.toml
├── src
│   ├── download_data.py
│   ├── evaluate.py
│   ├── train_speed.py
│   ├── train_tempo.py
│   └── utils.py
└── uv.lock                     # UV environment lockfile
│
```

## Dataset

The training and evaluation dataset is extracted from a single female speaker from the CMU Arctic Speech Database ([Kominek and Black, 2004](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.pdf)). 

Each sample is a 1 second utterance that have their speed and tempo modified by :  
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

0. 


The setting for these tests could be modified from the yaml files in `config/`

1. **Speed Model**  
The `train_speed` script trains a single SVM classifier.  
Pipeline :
    - Frame averaging
    - Feature scaling
    - PCA (Optional)
    - SVM Classifer

2. **Tempo Model**  
The `train_tempo` script trains an ensemble of 5 SVM classifiers, each classifier is trained on different frequency range, with the last model being trained on the full filterbank features.  
Pipeline :
    - Select frequency band
    - Average absolute difference (Rate of change)
    - Feature scaling
    - PCA (On the full band model)
    - SVM Classifier
    - Voting Classifier

Frequency band of each SVM classifier:

| Model | Frequency Channels   |
|-------|----------------------|
| 1     | 0 – 15               |
| 2     | 16 – 31              |
| 3     | 32 – 47              |
| 4     | 48 – 63              |
| 5     | 0 – 63               |

## Results




## Sources

This project use data from
- CMU Arctic Speech Database [Kominek and Black, 2004](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.pdf)
