import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import gaussian
import time

# Define function to load model based on user selection
def load_model(model_name):
    model_path = "C://aditi/competitions/aistudent/Metal_Corrosion/random_forest_model88.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            progress = st.sidebar.progress(0)
            for i in range(100):
                time.sleep(0.01)  # Simulating loading time
                progress.progress(i + 1)
            model = pickle.load(f)
            st.sidebar.text(f"Loaded {model_name} model successfully!")
            return model
    else:
        st.sidebar.error("Model file not found!")
        return None

# Load labels
labels = ["No Defect", "Metal corrosion"]

# Set page configuration
st.set_page_config(
    page_title="Metal Corrosion Detection",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛠 Metal Corrosion Detection")
st.subheader("🔍 Discovering Imperfections: A Machine Learning Approach")

# Add a sidebar for navigation
with st.sidebar:
    page = st.radio("Navigate", ["🏠 Home"])
    model_name = "Random Forest"
    model = load_model(model_name)

# Display page content
""" if page == "📚 About":
    st.write("Welcome to our Metal Corrosion Detection website!")
    st.write("Our mission is to provide an innovative solution for detecting corrosion in metal, addressing the challenges faced by industries in maintaining quality standards and ensuring safety.")
    st.write("Key features of our project:")
    st.write("- Utilizes machine learning algorithms to detect metal corrosion with accuracy.")
    st.write("- Detects Metal corrosion in areas inaccessible to human inspection, enhancing safety measures.")
    st.write("- Employs a self-made dataset tailored to reflect the corrosion in different types of metals.")
    st.write("- Highlights the importance of metal integrity in various industries and the potential hazards posed by undetected corrosion.")

elif page == "Metals Overview":
    st.write("### Metals Overview")
    st.write("Here's a comparison of different metals and their importance in various industries:")
    #st.image("output.png", use_column_width=True)
    st.write("#### Bronze:")
    st.write("Bronze is a versatile and popular metal used in various industries thanks to its properties, including corrosion resistance and low friction levels. It’s used in everyday objects to grand architectural masterpieces, electrical connectors, and bearings.")

    st.write("#### Iron:")
    st.write("Iron is used for making automobiles, trains, ships, engines, tools, and many other items. It is also used in construction due to its strength and ability to withstand extreme temperatures and weather conditions.")

    st.write("#### Steel:")
    st.write("Steel plays a vital role in the modern world. In addition to being one of the most important materials for building and infrastructure, steel is the enabler of a wide range of manufacturing activities.")

    st.write("#### Stainless Steel:")
    st.write("Stainless steel is primarily made from medium and low-carbon steel. They are alloyed with a range of metals to alter the resulting properties. For example, chromium and nickel lend corrosion resistance and hardness.")

    st.write("#### Aluminum:")
    st.write("Aluminum is a lightweight, corrosion-resistant, highly malleable, and infinitely recyclable material which finds usage in multiple industries, including construction, transport, electrical equipment, machinery, and packaging.")

    st.write("#### Copper:")
    st.write("Copper is an incredibly versatile mineral and its properties – high flexibility, conformity, thermal & electrical conductivity, and resistance to corrosion – make it critical to our domestic manufacturing sector.")

    st.write("#### Nickel:")
    st.write("Nickel, with its remarkable properties and rich history, has established itself as a vital metal in various sectors. Its corrosion resistance, high melting point, mechanical strength, and conductivity make it an indispensable element in applications ranging from stainless steel production to aerospace engineering.")
 """
 

# Home page: Upload and classify an image
uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        st.write("Classifying...")
        # Process the image
        image_array = np.array(image.convert('RGB'))
        image_resized = cv2.resize(image_array, (256, 256))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        #image_blurred = gaussian(image_gray, sigma=1.0)
        image_flattened = image_gray.flatten().reshape(1, -1)

        if model:
            prediction = model.predict(image_flattened)
            st.success(f"This metal surface has: {labels[prediction[0]]}")
        else:
            st.error("Failed to load model, please select a model.")

#@st.cache

# Footer with team members' names
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; right: 10px;">
        <p style="font-size: 12px;">Made by Aditi Dhavale</p>
    </div>
    """,
    unsafe_allow_html=True
)
