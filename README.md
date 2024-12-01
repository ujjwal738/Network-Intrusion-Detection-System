# Network-Intrusion-Detection-System

This repository contains a Python-based machine learning model designed to detect and classify network intrusions using the UNSW-NB15 dataset. The system leverages contrastive learning to train an embedded network that effectively distinguishes between normal and malicious traffic.

Table of Contents

Introduction
Features
Technologies Used
Setup Instructions
Usage
Model Architecture
Performance Metrics
Contributing
License

Introduction
Network security is crucial for safeguarding sensitive data and preventing malicious attacks. This project implements a Network Intrusion Detection System (NIDS) that identifies anomalies in network traffic using machine learning techniques.

The system employs a hybrid approach, combining supervised learning with contrastive learning to maximize detection accuracy and adaptability.

Features
Contrastive Learning: Leverages InfoNCE loss for enhanced feature representation.
Deep Embedding Network: Trains a multi-layer perceptron for learning latent representations.
One-Hot Encoding and Standard Scaling: Preprocesses categorical and numerical data for model training.
Real-Time Anomaly Classification: Identifies attack types in network traffic.
Performance Visualization: Provides insights into feature correlations and classification results using heatmaps and boxplots.
Technologies Used
Programming Languages: Python
Libraries and Frameworks:
NumPy, Pandas, Matplotlib, Seaborn
PyTorch (for deep learning models)
Scikit-learn (for preprocessing and evaluation)
Dataset: UNSW-NB15
Setup Instructions
Prerequisites
Python 3.8 or above
Virtual environment tools like venv or conda
Steps
Clone the Repository

git clone https://github.com/your-username/network-intrusion-detection.git
cd network-intrusion-detection
Install Dependencies

pip install -r requirements.txt
Prepare the Dataset
Download the UNSW-NB15 dataset and place it in the appropriate directory.

Run the Training Script

python train_model.py
Test the Model

python test_model.py
Usage
Training: Train the model with the training set from the UNSW-NB15 dataset.
Testing: Evaluate the model using the test set and visualize results.
Custom Classification: Use the pre-trained model to classify real-time or custom network traffic samples.
Model Architecture
The model consists of:

Input Layer: Processes encoded and scaled features.
Three Hidden Layers: Feature extraction using ReLU activation functions.
Embedding Layer: Generates 256-dimensional latent representations.
The model is optimized using the Adam optimizer and a step-based learning rate scheduler.

Performance Metrics
Training Loss: InfoNCE-based contrastive loss.
Test Accuracy: Evaluates detection performance on unseen data.
Visualization Tools: Heatmaps for feature correlation and classification reports.
Example Metrics:

Accuracy: 94.58%
Confusion Matrix: Evaluates true positives, false positives, etc.
Contributing
Contributions are welcome! Follow these steps:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit your changes (git commit -m "Add feature-name").
Push to the branch (git push origin feature-name).
Open a pull request.

