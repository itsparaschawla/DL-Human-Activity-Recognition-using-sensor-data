# Human Activity Recognition using Deep Learning on Sensor Data

This project implements a deep learning-based approach to recognize human activities using the UCI HAR dataset collected from smartphone inertial sensors (accelerometer and gyroscope). The model leverages a hybrid architecture combining 1D Convolutional Neural Networks (CNNs) and Bidirectional LSTM layers to extract both spatial and temporal features from the multi-channel time-series data. The workflow includes data preprocessing, normalization, model training, evaluation using confusion matrices and classification reports, and visualization of accuracy over epochs. This notebook demonstrates effective use of deep learning for real-world activity recognition tasks.

## 📊 Dataset

- **Name**: UCI HAR Dataset  
- **Source**: [UCI HAR DATASET](https://www.kaggle.com/datasets/drsaeedmohsen/ucihar-dataset/data)
- **Description**: Collected from 30 subjects performing daily activities like walking, standing, and sitting while wearing a smartphone on the waist. Data includes 3-axial linear acceleration and angular velocity captured at 50Hz.

## 🧠 Model Architecture

- 1D Convolution layers for feature extraction
- Bidirectional LSTM layers for capturing temporal dependencies
- Fully connected dense layers for final classification

## 🛠️ Requirements

Make sure to install the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
