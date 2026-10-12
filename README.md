# LieDetector-AI

Experimental AI project for detecting deceptive behavior using facial cues and machine learning.

This project analyzes face landmarks in real time to estimate behavioral indicators such as eye openness, blink variation, and mouth movement. It is built with Python, OpenCV, MediaPipe, and scikit-learn.

This project is intended for educational and research purposes only and is not a reliable lie detector.

make the dataset.

# How to Run

1: Collect DataRun the collection script to capture facial metrics.
Press v for Truth or m for Lie to label frames and save them to dataset.csv.
````Bash
python main.py

````
2: Train the ModelTrain a Random Forest classifier on your collected dataset: 
````Bash
python train.py

`````
3: Real-Time DetectionRun the detector to view real-time predictions based on your trained model[cite: 2]:
````Bash
   python detector.py

````
5: Playful Gauge (Optional)Run the standalone baseline-based variation gauge:
````Bash
python detecteur_ludique.py

````
## Run
```bash
pip install opencv-python mediapipe numpy scikit-learn pandas joblib
python main.py
python train.py
python detector.py


