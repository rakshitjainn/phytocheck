import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomBrightness
from tensorflow.keras.utils import to_categorical

DATASET_PATH = 'dataset'
IMG_SIZE_CLASSICAL = 64
IMG_SIZE_CNN = 128

def load_data(img_size):
    X = []
    y = []
    classes = os.listdir(DATASET_PATH)
    for class_name in classes:
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_dir): continue
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(class_name)
                
    return np.array(X), np.array(y)

if __name__ == '__main__':
    print("Loading data for Classical ML...")
    X_class, y_raw = load_data(IMG_SIZE_CLASSICAL)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    X_flat = X_class.reshape(X_class.shape[0], -1) / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Applying PCA...")
    pca = PCA(n_components=0.95) 
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel='rbf', probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    print("\n--- Training Classical ML Models ---")
    results = {}
    for name, model in models.items():
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")

    print("\n--- Hyperparameter Tuning (Random Forest) ---")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=1)
    grid_search.fit(X_train_pca, y_train)
    print(f"Best RF Parameters: {grid_search.best_params_}")
    print(f"Optimized RF Accuracy: {grid_search.best_score_:.4f}")

    print("\nSaving Classical ML Assets")
    joblib.dump(models["Support Vector Machine"], 'phyto_svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pca, 'pca.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    print("\nTraining CNN Model")
    X_cnn, _ = load_data(IMG_SIZE_CNN)
    X_cnn = X_cnn / 255.0 
    y_cnn_cat = to_categorical(y_encoded)

    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn_cat, test_size=0.2, random_state=42)

    # cnn_model = Sequential([
    #     RandomFlip("horizontal_and_vertical", input_shape=(IMG_SIZE_CNN, IMG_SIZE_CNN, 3)),
    #     RandomRotation(0.2), 
    #     RandomZoom(0.2),     
    #     RandomBrightness(factor=0.2), 
    #     Conv2D(32, (3,3), activation='relu'),
    #     MaxPooling2D(2,2),
    #     Conv2D(64, (3,3), activation='relu'),
    #     MaxPooling2D(2,2),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     Dense(len(le.classes_), activation='softmax')
    # ])

    # cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # history = cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, validation_data=(X_test_cnn, y_test_cnn), verbose=1)

    #Transfer Learning Mobilenetv2

    y_train_labels = np.argmax(y_train_cnn, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))

    print("Configuring MobileNetV2 Base Model...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE_CNN, IMG_SIZE_CNN, 3))
    base_model.trainable = False 

    cnn_model = Sequential([
        RandomFlip("horizontal_and_vertical", input_shape=(IMG_SIZE_CNN, IMG_SIZE_CNN, 3)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(le.classes_), activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )

    print("Training Transfer Learning Model...")
    history = cnn_model.fit(
        X_train_cnn, y_train_cnn,
        epochs=15,
        validation_data=(X_test_cnn, y_test_cnn),
        class_weight=class_weights_dict,
        callbacks=[early_stopping],
        verbose=1
    )

    cnn_model.save('phyto_cnn_model.keras')
    np.save('classes.npy', le.classes_)

    print("\nCNN Evaluation")
    y_pred_cnn = np.argmax(cnn_model.predict(X_test_cnn), axis=1)
    y_true_cnn = np.argmax(y_test_cnn, axis=1)

    print("Classification Report:")
    print(classification_report(y_true_cnn, y_pred_cnn, target_names=le.classes_))

    plt.figure(figsize=(10, 5))
    plt.bar(results.keys(), results.values(), color='skyblue')
    plt.title('Classical ML Models Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')

    cm = confusion_matrix(y_true_cnn, y_pred_cnn)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('CNN Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png')

    print("Training complete! Models and plots saved.")