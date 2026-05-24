# MNIST Digit Classifier

Handwritten digit classification project built with neural networks using Python and TensorFlow.

This project was developed as part of the course **Neural Networks** at the **Faculty of Sciences, University of Novi Sad**.

## Overview

The goal of this project is to classify handwritten digits from **0 to 9** using a fully connected neural network trained on the MNIST dataset.

The model is trained on thousands of handwritten digit images and is able to predict the digit shown in new unseen images.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib

## Features

- Load and preprocess the MNIST dataset
- Normalize image data before training
- Build a deep neural network using Dense layers
- Prevent overfitting using Dropout and Early Stopping
- Evaluate model performance on test data
- Save trained model to disk
- Predict digits from custom `.png` images

## Dataset

This project uses the **MNIST handwritten digits dataset**, which contains:

- 60,000 training images
- 10,000 test images

Each image is:

```text
28 x 28 pixels
