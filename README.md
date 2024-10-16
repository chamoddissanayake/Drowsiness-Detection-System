
# Drowsiness Detection System


The Drowsiness Detection System utilizes computer vision and deep learning to monitor driver alertness in real-time. It employs Haar cascades to detect faces and eyes, capturing eye states as "open" or "closed." A convolutional neural network (CNN) classifies these states, issuing alerts when prolonged eye closure is detected. If a driver shows signs of drowsiness, an alarm sounds, and the system visually indicates the alert status, enhancing road safety and reducing fatigue-related accidents.
## Run Locally

Install Python 3.10.0

https://www.python.org/downloads/release/python-3100/


Clone the project

```bash
  git clone https://github.com/chamoddissanayake/Drowsiness-Detection-System.git
```

Install dependencies

```bash
  pip install -r requirements.txt
```

If your device has multiple cameras, change camera id in here

```bash
  cap = cv2.VideoCapture(0)
```

Run the Application

```bash
  python app.py
```
## Tech Stack

**Programming Language:**
* Python
* Typescript

**Libraries and Frameworks:**

* OpenCV (for computer vision and video processing)
* Keras (for building and training neural networks)
* TensorFlow (backend for Keras)
* NumPy (for numerical operations)
* Pygame (for sound management)

**Machine Learning:**

* Convolutional Neural Networks (CNNs) for image classification

**Data Preprocessing:**

* ImageDataGenerator (for data augmentation)

**Model Saving and Loading:**

  * HDF5 (for saving Keras models)

**User Interface:**

  * OpenCV GUI (for video display)

**Hardware:**
  * Webcam (for real-time video capture)