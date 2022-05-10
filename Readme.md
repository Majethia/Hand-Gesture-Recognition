# Hand Gesture Detection

## Introduction:
A Program code to detect hands and the gestures. this is coded in Python using SKlearn and Mediapipe.

## Gestures recognized:
- Lost
- Victory
- Promiste
- Fist
- Wait
- Rock
- Call me
- One Minute

## Images Dataset:
The images dataset was taken from a [kaggle](https://www.kaggle.com/datasets/bikashpandey17/hand-sign-recognition) post.


The data1 and data2 csv file was generated using the `static.py` file and the images from the dataset.

## Training Models Used:

### Using KNN:
The model trained using KNN gave us a accuracy of `0.99921875`
this model is saved in `best.pikle` file.

### Using SVM:
The model trained using SVM gave us a accuracy of `1.0`
this model is saved in `best1.pikle` file.

## Final Product:
The final product takes input from webcam and gives realtime predictions for the handgestures using the SVM model saved in `best1.pikle` file.

## Run this on your own machine:
- First clone the repository using the following command
  `git clone https://github.com/Majethia/Hand-Gesture-Recognition`
  or if you dont have git cli, download the zip file and extract in a folder.
- Next, install the requirements
  `pip install -r requirements.txt`
- Run the main.py file
  `python main.py`
