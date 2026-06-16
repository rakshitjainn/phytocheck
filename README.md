# 🌿 PhytoCheck: Automated Plant Disease Diagnostic System

An end-to-end computer vision and machine learning pipeline engineered to classify 15 distinct plant diseases across a highly imbalanced dataset of 20,000+ agricultural images. 

This project provides real-time disease diagnosis and actionable agricultural treatment recommendations through a low-latency web interface.

## ✨ Core Features
* **Deep Learning Architecture:** Utilizes Transfer Learning via Google's pre-trained MobileNetV2 to extract complex visual features while bypassing hardware constraints.
* **Classical ML Benchmarking:** Features a comparative baseline Support Vector Machine (SVM) pipeline built with Principal Component Analysis (PCA) for 95% variance dimensionality reduction.
* **Algorithmic Balancing:** Implements dynamic class-weighting and early-stopping callbacks to successfully mitigate severe dataset imbalance and prevent mode collapse.
* **Interactive UI:** A fully deployed Streamlit frontend that translates raw tensor predictions into user-friendly confidence scores and agricultural treatment plans.

## 📊 Model Performance & Engineering
During development, extensive A/B testing was conducted to determine the optimal deployment model:
1. **Baseline Model (SVM):** A Support Vector Machine trained on flattened, PCA-reduced images achieved an **81% accuracy**.
2. **Optimized Model (MobileNetV2):** By replacing the shallow custom CNN with a pre-trained MobileNetV2 base and applying mathematical class weights, the deep learning pipeline outperformed the baseline, achieving **85% overall accuracy** with highly balanced precision/recall across minority classes.

## 🚀 Installation & Setup

### 1. Clone the Repository
    git clone https://github.com/rakshitjainn/phytocheck.git
    cd PhytoCheck
### 2. Create a Virtual Environment (Recommended)
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Mac/Linux:
    source venv/bin/activate
### 3. Install Dependencies
Ensure you have requirements.txt configured with streamlit, tensorflow, opencv-python, scikit-learn, numpy, pandas, and Pillow.
```
  pip install -r requirements.txt
```
4. Run the Application
The repository contains the pre-trained .keras model, meaning no manual training is required to run the frontend.
```
streamlit run app.py
```
## 📁 Project Structure
```text
PhytoCheck/
│
├── app.py                   # Streamlit frontend and prediction logic
├── train_pipeline.py        # Model training, A/B testing, and evaluation script
├── phyto_cnn_model.keras    # Trained MobileNetV2 Deep Learning Model
├── classes.npy              # Encoded target labels (15 diseases)
├── .gitignore               # Git rules to ignore raw datasets and caches
└── README.md                # Project documentation
```

*Developed by Rakshit Jain | Computer Science & Engineering Undergraduate.*
