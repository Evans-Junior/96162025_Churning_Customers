# Customer Churning Prediction App

## Overview

This project is a Streamlit web application designed to predict customer churn based on various input features. It utilizes a machine learning model trained on customer data, allowing users to input information and receive predictions about whether a customer is likely to churn.

## Features

- **User-Friendly Interface:** The Streamlit app provides a simple and intuitive interface for users to input customer data.
- **Churn Prediction:** The app uses a pre-trained machine learning model to predict whether a customer is likely to churn.
- **Confidence Factor:** The application provides a confidence factor, indicating the model's certainty in its prediction.

## How to Use

1. Install the required dependencies using the provided `requirements.txt` file.
2. Run the Streamlit app by executing the command `streamlit run your_app_filename.py`.
3. Fill out the form with customer information.
4. Click the "Predict Churn" button to see the model's prediction.
5. The result will display whether the customer is likely to churn and the confidence factor.

## Model Details

- **Model Type:** Keras Neural Network using a Functional API
- **Activation Functions:** Relu and hyperbolic tangent.
- **Output Layer:** Sigmoid.
- **Training:** The model is trained using a combination of scaled features and encoded categorical inputs.

## Note

- The confidence factor is calculated based on the probability of the predicted class.
- The app is designed to help businesses anticipate customer churn and take proactive measures.

### Link to video
# https://drive.google.com/file/d/1ESPs6vPWzZ_41KmS6v43NyuR2xQW1H8O/view?usp=sharing
