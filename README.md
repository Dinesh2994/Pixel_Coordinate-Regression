# ML Assignment – Supervised Regression: Predicting Pixel Coordinates

## Overview
This project demonstrates a deep learning approach to predict the coordinates `(x, y)` of a single bright pixel in a grayscale image.  
Each input image is `50x50` pixels, with all pixels set to 0 except one pixel with a value of 255, randomly located. The goal is to train a model that can accurately predict the coordinates of this bright pixel.

## Problem Statement
- Input: Grayscale image of size `50x50` pixels.
- Output: `(x, y)` coordinates of the pixel with value 255.
- Task: Build a supervised regression model using deep learning techniques.

## Dataset
- The dataset consists of 1000 images in PNG format, stored in Google Drive (`/content/drive/MyDrive/DeepEdgeAI_ML_Assignment`).
- Each image contains exactly one white pixel (value 255) and the rest are black (value 0).
- Coordinates of the white pixel are extracted and stored as ground-truth labels.

### Rationale for Dataset
- Small grayscale images allow faster training and lower computational cost.
- The single-pixel target introduces sparsity, making it a challenging regression task.
- Random placement ensures model generalization for any pixel location.

## Data Preprocessing
- Images are read in grayscale mode using OpenCV.
- Pixel values normalized to `[0, 1]`.
- Coordinates of the white pixel extracted using NumPy.
- Images reshaped to `(N, 50, 50, 1)` for CNN compatibility.
