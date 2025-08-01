#hello this is deep
this is an upgrade from the previous project i worked on
-uses mediapipe and cnn 
## HERE IS THE LINK FOR THE DATASET I USED: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
(paste into a folder named dataset. Copy the contents of extracted files into the main project folder.
directory goes like
-dataset
  -a
  -b
  -c
  -etc...


# Real-Time Sign Language Recognition



This project recognizes sign language letters from a webcam in real time.  
It uses a trained CNN model, OpenCV for video capture, and MediaPipe for hand detection.  
The recognized letters are shown on the screen and can form words. There is also an option to convert the text to speech.

## Features

- Detects hand signs from a webcam  
- Converts gestures to letters and builds words  
- Supports space, delete, and nothing gestures  
- Can speak the text using text-to-speech  

## Requirements

- Python 3.8 or newer  
- opencv-python  
- mediapipe  
- tensorflow  
- pyttsx3  
- numpy  

Install all dependencies with:

```
pip install opencv-python mediapipe tensorflow pyttsx3 numpy
```

## Project Files

- dataset/ : ASL Alphabet dataset  
- model/sign_model.h5 : Trained model  
- train_model.py : Script to train the model  
- predict.py : Script to run the webcam recognition  

## How to Run

1. Make sure the trained model file `sign_model.h5` is in the model folder.  
2. Run the prediction script:

```
python predict.py
```

3. The window will show the webcam feed and the recognized letter.  
4. Press S to speak the text, C to clear it, and Q to quit.
