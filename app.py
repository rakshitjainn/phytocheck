import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="PhytoCheck - Plant Disease AI", page_icon="🌿", layout="centered")

RECOMMENDATIONS = {
    "Pepper__bell___Bacterial_spot": {"disease": "Pepper Bell: Bacterial Spot", "treatment": "Use copper-based bactericides. Ensure seeds are disease-free and practice crop rotation."},
    "Pepper__bell___healthy": {"disease": "Pepper Bell: Healthy", "treatment": "Plant is healthy. Maintain proper watering and nutrition."},
    "Potato___Early_blight": {"disease": "Potato: Early Blight", "treatment": "Apply fungicides containing chlorothalonil or copper. Practice crop rotation and ensure good plant spacing for airflow."},
    "Potato___healthy": {"disease": "Potato: Healthy Plant", "treatment": "Your plant looks healthy! Continue regular watering, sunlight exposure, and standard fertilization."},
    "Potato___Late_blight": {"disease": "Potato: Late Blight", "treatment": "Immediate application of protective fungicides. Destroy infected plants to prevent spreading. Avoid overhead watering."},
    "Tomato__Target_Spot": {"disease": "Tomato: Target Spot", "treatment": "Apply fungicides like chlorothalonil. Remove infected lower leaves to improve air circulation."},
    "Tomato__Tomato_mosaic_virus": {"disease": "Tomato: Mosaic Virus", "treatment": "No cure exists. Remove and destroy infected plants immediately. Wash hands and tools thoroughly."},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"disease": "Tomato: Yellow Leaf Curl Virus", "treatment": "Control whitefly populations using insecticidal soaps or neem oil. Remove infected plants."},
    "Tomato_Bacterial_spot": {"disease": "Tomato: Bacterial Spot", "treatment": "Apply copper fungicides mixed with mancozeb. Avoid overhead irrigation."},
    "Tomato_Early_blight": {"disease": "Tomato: Early Blight", "treatment": "Use fungicides with chlorothalonil. Stake plants to keep foliage off the soil."},
    "Tomato_healthy": {"disease": "Tomato: Healthy", "treatment": "Plant is healthy. Continue standard care."},
    "Tomato_Late_blight": {"disease": "Tomato: Late Blight", "treatment": "Apply specific fungicides like mefenoxam if resistant strains are absent. Destroy severely infected plants."},
    "Tomato_Leaf_Mold": {"disease": "Tomato: Leaf Mold", "treatment": "Increase air circulation and reduce humidity. Apply approved fungicides if necessary."},
    "Tomato_Septoria_leaf_spot": {"disease": "Tomato: Septoria Leaf Spot", "treatment": "Apply chlorothalonil or copper fungicides. Remove infected lower leaves and practice crop rotation."},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"disease": "Tomato: Two-Spotted Spider Mite", "treatment": "Use insecticidal soaps, neem oil, or specific miticides. Introduce predatory mites as biological control."}
}

@st.cache_resource
def load_deep_learning_assets():
    try:
        # Loading the modern .keras format
        model = tf.keras.models.load_model('phyto_cnn_model.keras')
        classes = np.load('classes.npy')
        return model, classes
    except Exception as e:
        st.error(f"Error loading model. Please ensure the training script has finished. Details: {e}")
        return None, None

model, classes = load_deep_learning_assets()

st.title("🌿 PhytoCheck")
st.subheader("Plant Disease Detection & Treatment Recommendation")
st.markdown("Upload an image of a plant leaf, and our MobileNetV2 Deep Learning model will identify the disease and provide treatment recommendations.")

uploaded_file = st.file_uploader("Upload a Leaf Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    if st.button("Analyze Image"):
        if model is None:
            st.error("Model is not available.")
        else:
            with st.spinner("Analyzing via Transfer Learning Pipeline..."):
                img_array = np.array(image.convert('RGB'))
                
                # CNN uses 128x128 as defined in your script
                img_resized = cv2.resize(img_array, (128, 128))
                img_normalized = img_resized / 255.0
                img_batch = np.expand_dims(img_normalized, axis=0)
                
                # Predict
                predictions = model.predict(img_batch)
                confidence = np.max(predictions[0])
                predicted_class_idx = np.argmax(predictions[0])
                
                predicted_class_name = classes[predicted_class_idx]
                
                info = RECOMMENDATIONS.get(predicted_class_name, {
                    "disease": predicted_class_name.replace("___", " ").replace("_", " "),
                    "treatment": "Consult a local agricultural expert for specific treatment plans."
                })
                
                st.markdown("---")
                st.header("Diagnosis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Predicted Disease", value=info["disease"])
                with col2:
                    st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                
                st.subheader("📝 Treatment Recommendations")
                st.info(info["treatment"])
                
                st.progress(float(confidence))

st.markdown("---")
st.caption("Developed by a CSE Undergrad | Built with TensorFlow & Streamlit")