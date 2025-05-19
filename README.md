# Handwritten Text Recognition

This project implements a Handwritten Text Recognition (HTR) system using a Convolutional Recurrent Neural Network (CRNN) model. The model is built with TensorFlow and Keras. It's trained on the IAM Handwriting Database and uses Connectionist Temporal Classification (CTC) loss for sequence-to-sequence learning.

## Project Overview

The primary goal of this project is to accurately transcribe handwritten text from images into digital text. This is achieved by:

1.  **Data Preprocessing**: Images are read, decoded from PNG format, and resized while preserving aspect ratio. Padding is added to ensure uniform image dimensions. The images are then normalized. Labels (the actual text) are vectorized into numerical representations.
2.  **Model Architecture**: The CRNN model consists of:
    * Convolutional layers for feature extraction from the images.
    * Max-pooling layers to reduce dimensionality.
    * Recurrent Neural Network (RNN) layers (Bidirectional LSTMs) to capture sequential information in the text.
    * A Dense layer with softmax activation to output character probabilities.
    * A custom CTC Layer to calculate the CTC loss during training.
3.  **Training**: The model is trained using the Adam optimizer and CTC loss. An `EditDistanceCallback` is used to monitor the mean edit distance on a validation set at the end of each epoch, providing insights into the model's performance.
4.  **Prediction/Decoding**: After training, the model predicts character sequences from images. A utility function `decode_batch_predictions` uses a greedy search (or optionally beam search for more complex tasks) through the CTC decoded output to convert the numerical predictions back into human-readable text.
5.  **Optional Text Correction**: The notebook also includes a function `ndecode_batch_predictions` which incorporates a spelling correction mechanism using a vocabulary built from a text file (`big.txt`). This aims to improve the accuracy of the predicted text by correcting common spelling errors.

## Dataset

The project utilizes the IAM Handwriting Database, specifically the "words.txt" file for image paths and corresponding labels, and the "words" directory containing the actual images of handwritten words.

The data is split as follows:
* **Training samples**: 95% of the dataset (after filtering out errored entries).
* **Validation samples**: 50% of the remaining 5% (i.e., 2.5% of the total).
* **Test samples**: The other 50% of the remaining 5% (i.e., 2.5% of the total).

## Model Details

* **Input Image Size**: 128x32 (width x height)
* **Convolutional Blocks**:
    * Conv1: 32 filters, (3,3) kernel, ReLU activation, HeNormal initializer, same padding.
    * Pool1: MaxPooling (2,2).
    * Conv2: 64 filters, (3,3) kernel, ReLU activation, HeNormal initializer, same padding.
    * Pool2: MaxPooling (2,2).
* **Reshaping**: The output from the convolutional blocks is reshaped before being fed to the RNN layers. The new shape is `(image_width // 4, (image_height // 4) * 64)`.
* **Dense Layer (before RNN)**: 64 units, ReLU activation.
* **Dropout (before RNN)**: 0.2
* **RNN Layers**:
    * Bidirectional LSTM: 128 units, return sequences, dropout 0.25.
    * Bidirectional LSTM: 64 units, return sequences, dropout 0.25.
* **Output Dense Layer**: `len(characters) + 2` units (vocabulary size + blank token + extra for safety), softmax activation.
* **Optimizer**: Adam
* **Loss Function**: CTC Loss (implemented in `CTCLayer`)

## Setup and How to Run

1.  **Dependencies**:
    * TensorFlow (Keras)
    * NumPy
    * Matplotlib
    * OpenCV (implied by image processing, though not explicitly imported as `cv2` in the provided snippet for `Main.ipynb`)
    * A text file named `big.txt` (containing a large corpus of text for spelling correction, e.g., from Norvig's spelling corrector) and `bigg.txt` (likely a subset or specific vocabulary for checking words) should be in the `/content/` directory.
    * The IAM dataset:
        * `data/words.txt`
        * `data/words/` (directory containing subdirectories of images)

2.  **Data Preparation**:
    * Ensure the `words.txt` file and the `words` image directory are in a subdirectory named `data/`.
    * Place `big.txt` and `bigg.txt` in the `/content/` directory if using the spelling correction feature.

3.  **Execution**:
    * The `Main.ipynb` notebook contains the complete workflow.
    * **Load Data**: The notebook first loads the image paths and labels from `words.txt`.
    * **Preprocessing**: It then preprocesses these images and labels, vectorizing the text and preparing TensorFlow datasets (`train_ds`, `validation_ds`, `test_ds`).
    * **Model Building**: The `build_model()` function defines the CRNN architecture.
    * **Training**: The model is compiled and trained using `model.fit()`. The `EditDistanceCallback` and `prediction_model` (a version of the model for inference) are defined and used during training.
    * **Inference & Visualization**: After training (or if loading a pre-trained model, which is not explicitly shown in this notebook but is a common practice), the `prediction_model` can be used to predict text from new images. The notebook includes a section to visualize some test images along with their original labels and the model's predictions.
    * **Evaluation**: The `accuracy_on_epoch` and `accuracy_on_epoch2` lists store the accuracy on the test set after each epoch, with and without the dictionary-based correction, respectively. These are plotted at the end.

## Key Functions

* `decode_batch_predictions(pred)`: Decodes the model's raw predictions into text using greedy search.
* `ndecode_batch_predictions(pred)`: Decodes predictions and applies spelling correction.
* `get_image_paths_and_labels(samples)`: Retrieves image paths and their corresponding text labels.
* `clean_labels(labels)`: Extracts the actual text from the label strings.
* `distortion_free_resize(image, img_size)`: Resizes images without distortion.
* `preprocess_image(image_path, img_size)`: Reads, decodes, resizes, and normalizes an image.
* `vectorize_label(label)`: Converts text labels to numerical sequences.
* `process_images_labels(image_path, label)`: Applies preprocessing to an image-label pair.
* `prepare_dataset(image_paths, labels)`: Creates a TensorFlow dataset from image paths and labels.
* `CTCLayer(keras.layers.Layer)`: Custom Keras layer for CTC loss calculation.
* `build_model()`: Defines and compiles the CRNN model.
* `calculate_edit_distance(labels, predictions)`: Computes the edit distance between true labels and predictions.
* `EditDistanceCallback(keras.callbacks.Callback)`: Keras callback to calculate and log edit distance and accuracy during training.
* `correction(word)`: Spelling correction function.

## Results

The notebook plots the accuracy over epochs, comparing performance with and without the dictionary-based spelling correction. It also visualizes predictions on sample test images.
