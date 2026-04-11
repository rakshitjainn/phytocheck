PhytoCheck: Plant Disease Detection Using Machine Learning

Author: CSE Undergraduate
Domain: Machine Learning / Computer Vision

1. Project Overview

Problem Statement: Plant diseases threaten global food security and agricultural economy. Manual identification is slow, prone to error, and requires expert knowledge.
Solution: PhytoCheck is an automated Machine Learning system that classifies plant diseases from leaf images. It not only predicts the disease but also provides a confidence score and actionable treatment recommendations.
Relevance: This is a real-world application of Computer Vision and supervised learning, demonstrating a complete end-to-end ML pipeline.

2. Project Architecture (ML Workflow)

The project strictly follows the standard Machine Learning Workflow:

Data Collection: Sourcing leaf images from the PlantVillage dataset.

Data Preprocessing: Resizing images, normalizing pixel values, and flattening arrays.

Feature Extraction & Scaling: Using StandardScaler and Dimensionality Reduction (PCA) to extract dominant features for classical ML models.

Model Training: Training baseline classical models (Logistic Regression, KNN, SVM, Decision Trees, Random Forest) and an advanced Neural Network (CNN).

Model Evaluation: Comparing models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.

Optimization: Applying Hyperparameter tuning (GridSearchCV) to improve the best baseline model.

Deployment: Serving the best model (CNN) via a Streamlit Web App.

3. Dataset

Name: PlantVillage Dataset (Available on Kaggle)

Download Link: PlantVillage Dataset on Kaggle

Structure: The dataset contains folders, where each folder represents a class (e.g., Potato___Early_blight, Potato___Late_blight, Potato___healthy).

Note for implementation: For a smoother local execution, it is recommended to select a subset of 3-5 classes (e.g., just Potato or Tomato leaves) to prevent RAM overload during classical ML training.

6. Neural Network Model (Unit IV Theory)

While classical models rely on flattened pixels and PCA, we use a Convolutional Neural Network (CNN) for optimal image classification.

Architecture: Convolutional Layers -> MaxPooling -> Flatten -> Dense Layers.

Activation Functions: We use ReLU (Rectified Linear Unit) for hidden layers to introduce non-linearity and avoid the vanishing gradient problem. We use Softmax in the output layer to convert raw logits into class probabilities.

Cost Function: categorical_crossentropy is used as it calculates the loss between the predicted probability distribution and the actual one-hot encoded labels.

Gradient Descent & Backpropagation: We use the Adam optimizer (an advanced form of Gradient Descent). During training, the network makes a prediction (forward pass), calculates the loss, and then computes gradients backwards through the network (backpropagation) to update weights and minimize the error.

9. Disease Recommendation System

Once the model predicts a class, PhytoCheck maps it to a treatment knowledge base.

Example: If predicted Potato_Early_Blight:

Diagnosis: Early Blight

Recommendation: Apply copper-based fungicides. Ensure adequate spacing between plants for airflow. Remove infected leaves.

12. Project Structure

To impress your evaluator, maintain this professional folder structure:

PhytoCheck/
│
├── dataset/                     # Download PlantVillage and place folders here
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___healthy/
│
├── models/                      # Saved trained models (.pkl, .h5)
├── train_pipeline.py            # Main ML training script
├── app.py                       # Streamlit web app
├── requirements.txt             # Dependencies
└── PhytoCheck_Report.md         # This document


13. Viva Preparation (Q&A)

Q1: Why did you use PCA before applying SVM and Random Forest?
A: Images are high-dimensional data (e.g., a 64x64 RGB image has 12,288 features). Feeding this directly into classical models like SVM causes the "Curse of Dimensionality," making training extremely slow and prone to overfitting. PCA (Principal Component Analysis) reduces dimensionality by projecting data onto orthogonal axes that maximize variance, retaining ~95% of the information while drastically reducing the number of features.

Q2: Can you explain the evaluation metrics you used?
A: * Accuracy: Overall correctness of the model.

Precision: Out of all cases predicted as "Early Blight", how many actually were? (Crucial if false positives are costly).

Recall: Out of all actual "Early Blight" cases, how many did we detect? (Crucial if missing a disease destroys the crop).

F1-Score: The harmonic mean of Precision and Recall.

Confusion Matrix: A table showing True Positives, True Negatives, False Positives, and False Negatives.

ROC-AUC: Measures the model's ability to distinguish between classes at various threshold settings.

Q3: Why did the CNN perform better than Logistic Regression or KNN?
A: Classical models (LR, KNN) treat images as 1D arrays of isolated pixels, losing all spatial information (how pixels relate to their neighbors). CNNs use Convolutional filters to automatically extract spatial features (edges, textures, shapes) and build a hierarchy of patterns, making them vastly superior for image data.

Q4: How would you improve this system in the future?
A: 1.  Data Augmentation: Rotating, flipping, and zooming images to make the model robust to different lighting and angles.
2.  Transfer Learning: Using pre-trained models like ResNet50 or MobileNetV2 for higher accuracy.
3.  Deployment: Converting the model to TensorFlow Lite and building a mobile app so farmers can take pictures directly in the field.