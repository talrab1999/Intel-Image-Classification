# Intel Image Classification

## Overview
This project is an image classification model built using **PyTorch** to categorize images into six classes: **buildings, forest, glacier, mountain, sea, and street**. 
The dataset is sourced from **Kaggle's Intel Image Classification dataset** and consists of about 25,000 images.

## Dataset
The dataset is organized into three directories:
- `seg_train/` - Contains labeled training images for each class.
- `seg_test/` - Contains labeled test images for evaluation.
- `seg_pred/` - Contains unlabeled images for inference.

## Model
This project implements a **Convolutional Neural Network (CNN) from scratch**, meaning no pre-trained models were used. The model was designed and trained entirely from the ground up. 
This approach allows full control over the model's layers, parameters, and optimization process, rather than relying on transfer learning or pre-trained CNNs.

### Model Architecture
- **Conv2D layers** with ReLU activation
- **MaxPooling layers**
- **Fully connected (FC) layers**
- **Softmax activation** for classification

The model was built and fine-tuned manually to achieve strong classification performance without using any pre-trained networks.

## Training the Model
- The model is trained on the `seg_train` dataset.
- Performance is evaluated on `seg_test`.

## Results & Performance
Predictions include class probabilities before selecting the highest-scoring class.
