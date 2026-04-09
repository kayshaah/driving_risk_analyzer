# Intelligent Driving Behavior Risk Analyzer

The Intelligent Driving Behavior Risk Analyzer is a Python GUI application that evaluates driving patterns and classifies driver risk from motion-sensor data. It reads time-series driving data from CSV files, extracts driving-behavior features, calculates a risk score, and predicts a risk level using both rule-based analysis and a small machine-learning style classifier.



## Project structure

* 'main.py': main file to run the application
* 'driving\_risk\_analyzer/driving\_session.py': 'DrivingSession' summary class
* 'driving\_risk\_analyzer/analysis\_result.py': 'RiskAnalysisResult' class
* 'driving\_risk\_analyzer/dataset\_manager.py': CSV loading and validation
* 'driving\_risk\_analyzer/risk\_analyzer.py': feature extraction, heuristic scoring, and KNN-style prediction
* 'driving\_risk\_analyzer/driving\_risk\_app.py': tkinter GUI
* 'data/sample\_driving\_sensor\_data.csv': example sensor dataset
* 'data/risk\_reference\_profiles.csv': reference profiles used by the classifier
* 'tests/test\_risk\_analyzer.py': tests



## How to run

1. Install dependencies:
python3 -m pip install -r requirements.txt

2. Launch the app:
python3 main.py

## CSV sensor format

The driving sensor CSV must contain these columns:

* 'session\_id'
* 'timestamp\_sec'
* 'accel\_x'
* 'accel\_y'
* 'accel\_z'
* 'gyro\_x'
* 'gyro\_y'
* 'gyro\_z'
* 'speed\_kmh'

Each row represents one timestamped sensor reading for a driving session.

## How the program works

1. The user loads a CSV file containing time-series driving data.
2. The user chooses a driving session from the dropdown.
3. The app extracts features such as:

   * harsh acceleration events
   * harsh braking events
   * sharp turning events
   * swerving events
   * overspeed events
   * speed variability
4. The app computes a heuristic risk score.
5. The app also compares the extracted features against reference profiles using a custom KNN-style classifier.
6. The GUI displays the final label, confidence, extracted features, and safety recommendations.

## 

## Running tests

python3 -m pytest

## Expansion ideas (Future Scope)

* Add live sensor streaming
* Add chart visualizations for speed and turning behavior
* Add support for video analysis with OpenCV
* Allow exporting analysis results
* Train the classifier on a larger real-world dataset

