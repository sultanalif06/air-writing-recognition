# Air Writing Recognition

Air Writing Recognition is a computer vision and machine learning project that recognizes alphabet letters written in the air using hand landmark data.

## Features

* Collects hand movement data using a webcam
* Extracts hand landmarks using MediaPipe
* Preprocesses air writing trajectory data
* Trains a KNN model for letter classification
* Runs real-time inference using webcam input

## Project Structure

```txt
air-writing-recognition/
├── src/
│   ├── collect_data.py
│   ├── preprocess_dataset.py
│   ├── train_knn.py
│   └── realtime_inference.py
│
├── data/
│   └── raw/
│
├── models/
│   ├── knn_model_v2.pkl
│   └── label_encoder_v2.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone this repository:

```bash
git clone https://github.com/sultanalif06/air-writing-recognition.git
cd air-writing-recognition
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect Data

Run this script to collect air writing data using your webcam:

```bash
python src/collect_data.py
```

### 2. Preprocess Dataset

Run this script to preprocess the collected dataset:

```bash
python src/preprocess_dataset.py
```

### 3. Train Model

Run this script to train the KNN classification model:

```bash
python src/train_knn.py
```

### 4. Real-Time Inference

Run this script to recognize air writing input in real time:

```bash
python src/realtime_inference.py
```

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Scikit-learn
* Joblib

## Model

This project uses a K-Nearest Neighbors (KNN) model to classify air-written alphabet letters based on processed hand landmark trajectory data.

## Author

Sultan Alif Ibrahim Anwar
