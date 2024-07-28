# Predict_Social_Behavior
Predict social behavior based on electrophysiological data and other modalities
Deep Learning Model for Analyzing LFP Data from Rat Brains

This project aims to design a deep learning model to analyze raw Local Field Potential (LFP) data collected from rat brains during social behavior tests. The model is inspired by the Video Masked Autoencoder (VideoMAE) and utilizes Vision Transformers to learn a representation of the sequences of data.
Table of Contents

    Background
    Data Preprocessing
    Model Architecture
        Encoder
        Decoder
    Embedding Details
    Implementation Plan
        Data Collection and Preprocessing
        Model Construction
        Training and Evaluation
        Inference and Application
    Checklist

Background

The focus areas of the brain for this study are:

    Anterior Amygdala (AA)
    Medial Amygdala (MeA)
    Posterior Ventral Medial Amygdala (MePV)
    Central Amygdala (CeA)
    Stria Terminalis (STIA)

These areas are crucial for social behaviors, and their activity is significantly different between rats that exhibit social behaviors and those that do not.
Data Preprocessing

The raw LFP time-series data from the five brain areas will be converted into spectrograms. Options for spectrogram conversion include:

    Linear Spectrogram
    Mel Spectrogram
    Continuous Wavelet Transform (CWT)

The spectrograms will be combined into a 3D tensor and reshaped to have dimensions [time, area, frequency]. This method helps handle missing data points, as not all areas are measured in all rats.
Model Architecture

The model follows a structure similar to the Video Masked Autoencoder (VideoMAE), with an encoder and decoder setup.
Encoder

    Patchify Input: Use 3D convolutions with a kernel size of [2, x, y] to create patch embeddings.
    Positional Embedding: Incorporate time and area positional embeddings.
    Transformer Encoder: Encode the sequence using a Vision Transformer (ViT).

Decoder

    Fully Connected Layer: Process encoded representations.
    Transformer Decoder: Reconstruct the original masked patches.
    Reconstruction Objective: Train the model to predict the masked spectrogram patches accurately.

Embedding Details

    Time Embedding: Positional encoding for the time dimension.
    Area Embedding: Spatial encoding using the XYZ coordinates of each brain area.
    Class Embedding: Experimental stage embedding (baseline, encounter, after encounter).

Implementation Plan
Data Collection and Preprocessing

 Collect raw LFP data from the specified brain areas.
 Transform the time-series data into the chosen spectrogram format.

     Handle missing data appropriately during preprocessing.

Model Construction

 Design the patchification process using 3D convolutions.
 Implement positional, area, and class embeddings.

     Construct the encoder and decoder transformers based on the VideoMAE structure.

Training and Evaluation

 Train the model on the preprocessed dataset, ensuring to mask certain patches during training to simulate missing data.

     Evaluate the model’s performance in reconstructing masked patches and classifying social behaviors.

Inference and Application

 During inference, utilize the model’s ability to handle missing patches to predict outcomes even when some brain areas’ data are missing.

     Apply the trained model to new LFP datasets to identify social behavior patterns.

Checklist

 Data Collection and Preprocessing

 Collect raw LFP data from specified brain areas.
 Convert time-series data into spectrograms (linear, mel, or CWT).
 Combine spectrograms into a 3D tensor and reshape to [time, area, frequency].

     Handle missing data points appropriately.

 Model Construction

 Design patchification process using 3D convolutions.
 Implement positional, area, and class embeddings.

     Construct the encoder and decoder transformers.

 Training and Evaluation

 Train the model with masked patches to handle missing data.

     Evaluate model performance on reconstruction and classification tasks.

 Inference and Application

 Use the model to predict outcomes with missing data during inference.
 Apply the model to new LFP datasets to identify social behavior patterns.
