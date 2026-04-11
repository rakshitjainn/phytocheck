import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="PhytoCheck - Plant Disease AI", page_icon="🌿", layout="centered")

# Load Knowledge Base (Disease Recommendation System)
# In a real app, this could be a database.
RECOMMENDATIONS = {
    "Potato___Early_blight": {
        "disease": "Early Blight",
        "confidence_threshold": 0.75,
        "treatment": "Apply fungicides containing chlorothalonil or copper. Practice crop rotation and ensure good plant spacing for airflow."
    },
    "Potato___Late_blight": {
        "disease": "Late Blight",
        "confidence_threshold": 0.75,
        "treatment": "Immediate application of protective fungicides. Destroy infected plants to prevent spreading. Avoid overhead watering."
    },
    "Potato___healthy": {
        "disease": "Healthy Plant",
        "confidence_threshold": 0.50,
        "treatment": "Your plant looks healthy! Continue regular watering, sunlight exposure, and standard fertilization."
    }
}

# Load Model and Classes
@st.cache_resource
def load_model_and_classes():
    try:
        model = tf.keras.models.load_model('phyto_cnn_model.h5')
        classes = np.load('classes.npy')
        return model, classes
    except Exception as e:
        st.error(f"Error loading model. Please run train_pipeline.py first. Details: {e}")
        return None, None

model, classes = load_model_and_classes()

# Header
st.title("🌿 PhytoCheck")
st.subheader("Plant Disease Detection & Treatment Recommendation")
st.markdown("""
Upload an image of a plant leaf, and our Machine Learning model (CNN) will identify the disease and provide treatment recommendations.
""")

# File Uploader
uploaded_file = st.file_uploader("Upload a Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    if st.button("Analyze Image"):
        if model is None:
            st.error("Model is not available.")
        else:
            with st.spinner("Analyzing using Convolutional Neural Network..."):
                # Preprocess image for the CNN
                # Convert PIL image to numpy array
                img_array = np.array(image.convert('RGB'))
                
                # Resize to match CNN input shape (128x128)
                img_resized = cv2.resize(img_array, (128, 128))
                
                # Normalize pixel values
                img_normalized = img_resized / 255.0
                
                # Expand dimensions to match batch size format (1, 128, 128, 3)
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Predict
                predictions = model.predict(img_batch)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                predicted_class_name = classes[predicted_class_idx]
                
                # Fetch Recommendation
                # If the predicted class isn't exactly in our dummy dict, use a default fallback
                info = RECOMMENDATIONS.get(predicted_class_name, {
                    "disease": predicted_class_name.replace("___", " ").replace("_", " "),
                    "treatment": "Consult a local agricultural expert for specific treatment plans."
                })
                
                st.markdown("---")
                st.header("Diagnosis Results")
                
                # Layout columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Predicted Disease", value=info["disease"])
                with col2:
                    st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                
                st.subheader("📝 Treatment Recommendations")
                st.info(info["treatment"])
                
                # Progress bar for confidence visual
                st.progress(float(confidence))

st.markdown("---")
st.caption("Developed by a CSE Undergrad | Built with TensorFlow & Streamlit")