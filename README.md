# MNIST Handwritten Digit Classification with Keras and Streamlit

**Name : JUNAID AKBAR**

- 🏠 MIANWALI, PAKISTAN
- 📧 malikjuni70@gmail.com 
- 🔗 LINKhttps://www.linkedin.com/in/juanid-akbar/EDIN 
- 🔗 https://www.kaggle.com/junaiddata35   
- 🔗 https://github.com/Junaid-Akbar35

## SUMMARY
Powerful and Statistically sound and technically acute Data Science graduate with in-depth expertise in Machine
Learning, Deep Learning, NLP and Transfer Learning. Expertise in the area of developing end-to-end AI
applications using OpenAI, Gemini and DeepSeek APIs and prompt engineering of LLM. Practical experience
with Streamlit, FastAPI, Flask, Git, Docker, and Hugging Face Transformers, as well as ML pipelines ready to
be deployed. Excellent skill in feature engineering, data story telling and visual analytics through Power BI and
Tableau. Enthusiastic in applying AI in solving real-life problems.

## EDUCATION
University of  Mianwali,
Bachelor of Data Science

---

This project demonstrates an end-to-end deep learning workflow for classifying handwritten digits using the MNIST dataset. It covers key stages from data loading and preprocessing to model training with hyperparameter tuning, and finally deploys the trained model in an interactive web application built with Streamlit.

## Features

- **Data Loading and Visualization:** Load the MNIST dataset and visualize example images to understand the data.
- **Data Preprocessing and Augmentation:** Prepare the data for model training through normalization, reshaping, and applying data augmentation techniques to enhance the dataset's diversity.
- **Data Pipeline Optimization:** Optimize the data loading and preprocessing pipeline using `tf.data` for efficient training.
- **Hyperparameter Tuning:** Utilize Keras Tuner to perform hyperparameter search and identify the optimal configuration for the model.
- **CNN Model Building and Training:** Construct and train a Convolutional Neural Network model tailored for handwritten digit classification.
- **Model Evaluation:** Assess the trained model's performance on a separate test dataset.
- **Model Saving:** Save the trained model in a suitable format for deployment.
- **Streamlit Web Application:** Provide an interactive web interface built with Streamlit allowing users to upload images of digits and get predictions from the trained model.

## Getting Started

This section will guide you through setting up and running the project.

### Prerequisites

Make sure you have the following software installed on your system:

*   Python 3.7+
*   pip (Python package installer)

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project repository is organized as follows:

- `saved_models/`: Directory to store the trained Keras model.
- `logs/`: Directory containing TensorBoard logs from model training.
- `mnist_classification_notebook.ipynb`: The main Jupyter notebook detailing the end-to-end deep learning workflow.
- `streamlit_app.py`: The Python script for the Streamlit web application.
- `requirements.txt`: Lists the Python dependencies required to run the project.

## Model Details

The core of this project is a Convolutional Neural Network (CNN) designed for image classification. The architecture consists of convolutional layers for feature extraction, followed by pooling layers for dimensionality reduction. A flattening layer transitions to fully connected dense layers for classification. Dropout is included to prevent overfitting.

The model was trained on the widely-used MNIST dataset, which contains grayscale images of handwritten digits (0-9). Hyperparameter tuning was performed using the Keras Tuner library to find the optimal configuration for the CNN, resulting in improved performance.

## Streamlit Application

To run the interactive Streamlit application for handwritten digit classification, navigate to the root directory of the project in your terminal and execute the following command:

```bash
streamlit run streamlit_app.py
```

This command will start the Streamlit server and open the application in your default web browser. If it doesn't open automatically, copy and paste the local URL provided in the terminal output into your browser.

## Future Enhancements

Here are some potential improvements and future features for this project:

- Implement the drawing input functionality in the Streamlit application to allow users to draw digits directly.
- Add more robust error handling and user feedback in the Streamlit app for invalid inputs or processing issues.
- Explore more advanced data augmentation techniques.
- Experiment with different CNN architectures or other model types.
- Implement model deployment to a cloud platform (e.g., Heroku, AWS, GCP).
- Add a feature to visualize model performance metrics and loss curves within the Streamlit app.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or suggestions, feel free to reach out:

- **Name:** [Your Name]
- **GitHub:** [Your GitHub Profile Link]
- **Email:** [Your Email Address]
