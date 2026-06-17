\# Air Writing Recognition



Air Writing Recognition is a computer vision and machine learning project that recognizes alphabet letters written in the air using hand landmark data.



\## Features



\* Collect hand movement data using webcam

\* Extract hand landmarks using MediaPipe

\* Preprocess air writing trajectory data

\* Train a KNN model for letter classification

\* Run real-time inference using webcam



\## Project Structure



air-writing-recognition/



\* src/

\* data/raw/

\* models/

\* requirements.txt

\* README.md



\## Installation



Install the required dependencies:



pip install -r requirements.txt



\## Usage



Collect data:



python src/collect\_data.py



Preprocess dataset:



python src/preprocess\_dataset.py



Train model:



python src/train\_knn.py



Run real-time inference:



python src/realtime\_inference.py



