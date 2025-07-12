import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Add a markdown section at the beginning of the Streamlit code
st.markdown("""
# MNIST Handwritten Digit Classification App

This application allows you to classify handwritten digits using a deep learning model trained on the famous MNIST dataset. You can either upload an image of a digit or, in a future version, draw one directly in the app.
""")

# Include another markdown section to explain the deep learning model used
st.markdown("""
## About the Model

The model used in this application is a Convolutional Neural Network (CNN) specifically designed for image classification tasks. It was trained on the MNIST dataset, which consists of 70,000 grayscale images of handwritten digits (0-9). The CNN architecture is well-suited for recognizing patterns in images, making it effective for this task. The model has been fine-tuned using hyperparameter search to achieve high accuracy in classifying handwritten digits.
""")

# Define the path to the saved model file
model_path = "saved_models/mnist_cnn/mnist_model.keras"

# Load the trained Keras model
@st.cache_resource
def load_trained_model(model_path):
    """Loads the trained Keras model."""
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model
model = load_trained_model(model_path)

# Define global constants (these were defined in the training notebook)
# In a real app, you might load these from a config file or pass them
IMAGE_SIZE = (28, 28)
NUM_CLASSES = 10

st.write("Model and essential configurations loaded successfully!")

# Set the title of the Streamlit application
st.title("Handwritten Digit Classification")

# Create a section for image input
st.subheader("Provide a Handwritten Digit Image")

# Option 1: Upload an image
uploaded_file = st.file_uploader("Choose an image file...", type=["png", "jpg", "jpeg"])

# Option 2: Drawing input (Placeholder)
st.write("Drawing input is not available in this version.")

# Process the input image
processed_image = None

if uploaded_file is not None:
    try:
        # Open the uploaded image
        img = Image.open(uploaded_file)

        # Convert to grayscale
        img = img.convert('L')

        # Resize the image
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img)

        # Invert colors (MNIST is white digit on black background)
        img_array = 255 - img_array

        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0

        # Reshape to add batch and channel dimensions (1, 28, 28, 1)
        processed_image = np.reshape(img_array, (1, *IMAGE_SIZE, 1))

        st.write("Image processed successfully!")
        st.image(img, caption="Processed Image (Grayscale, Resized, Inverted)", use_column_width=True)


    except Exception as e:
        st.error(f"Error processing image: {e}")

# Make predictions
predictions = None

if processed_image is not None:
    # Use the loaded model to make a prediction
    predictions = model.predict(processed_image)
    st.write("Prediction made successfully!")

# Display the results
if predictions is not None:
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=1)[0]

    st.subheader("Prediction Result")
    st.write(f"The predicted digit is: **{predicted_class}**")

    # Display confidence scores
    st.subheader("Confidence Scores")
    # Create a dictionary for the bar chart
    confidence_dict = {str(i): predictions[0][i] for i in range(NUM_CLASSES)}
    st.bar_chart(confidence_dict)
