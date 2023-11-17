import streamlit as st
import pickle
from keras.models import load_model


def map_input_to_values(value, field):
    if field == 'InternetService':
        return {'DSL': 0, 'Fiber optic': 1, 'No': 2}.get(value)
    elif field == 'gender':
        return {'Female': 0, 'Male': 1}.get(value)
    elif field == 'PaymentMethod':
        return {'Electronic check': 2, 'Mailed check': 3, 'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1}.get(value)
    elif field == 'Contract':
        return {'Month-to-month': 0, 'One year': 1, 'Two year': 2}.get(value)
    elif field == 'TechSupport':
        return {'No': 0, 'Yes': 2, 'No internet service': 1}.get(value)
    elif field == 'OnlineBackup':
        return {'Yes': 2, 'No': 0, 'No internet service': 1}.get(value)
    elif field == 'OnlineSecurity':
        return {'No': 0, 'Yes': 2, 'No internet service': 1}.get(value)
    else:
        return None


# Load the Keras model
model = load_model('my_trained_keras_model.h5')
with open("util.pkl", "rb") as f:
    scaler= pickle.load(f)

def scale_features(tenure, monthly_charges, total_charges):
    scaled_features = scaler.transform([[tenure, monthly_charges, total_charges]])
    return scaled_features

def prediction(scaled_tenure, scaled_monthly_charges, scaled_total_charges, online_backup, online_security, tech_support, contract, payment_method, gender, internet_service):
# Create a feature vector combining encoded values and scaled features
    feature_values=[scaled_tenure, scaled_monthly_charges, scaled_total_charges, online_backup, online_security, tech_support, contract, payment_method, gender, internet_service]

    # Make a prediction
    predicted_value = model.predict([feature_values])    
    return predicted_value

    
def main():
    
    st.title('Customer Churning Prediction Form')

    st.write("Fill out the following form to predict customers' Churn")

    tenure = st.number_input('Tenure', value=0)
    monthly_charges = st.number_input('Monthly Charges', value=0.0)
    total_charges = st.number_input('Total Charges', value=0.0)

    online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    gender = st.selectbox('Gender', ['Female', 'Male'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    original_churn_result=st.selectbox('Original Churn Result', ['No', 'Yes'])
    # Converting to labeledEncoded format
    online_backup = map_input_to_values(online_backup, 'OnlineBackup')
    online_security = map_input_to_values(online_security, 'OnlineSecurity')
    tech_support = map_input_to_values(tech_support, 'TechSupport')
    contract = map_input_to_values(contract, 'Contract')
    payment_method = map_input_to_values(payment_method, 'PaymentMethod')
    gender = map_input_to_values(gender, 'gender')
    internet_service = map_input_to_values(internet_service, 'InternetService')

    if st.button('Predict Churn'):
        data = scale_features(tenure, monthly_charges, total_charges)
        print(data)
        scaled_tenure=data[0][0]
        scaled_monthly_charges=data[0][1]
        scaled_total_charges=data[0][2]
        predicted_value = prediction(scaled_tenure, scaled_monthly_charges, scaled_total_charges, online_backup, online_security, tech_support, contract, payment_method, gender, internet_service)
        # Convert the predicted integer value to 'No' or 'Yes'
        print(predicted_value)
        # Calculate the probability for the original class
        predicted_prob_yes = predicted_value if original_churn_result == 'Yes' else 1 - predicted_value

        # Calculate the probability for the opposite class
        predicted_prob_no = 1 - predicted_prob_yes

        # Calculate the confidence factor
        confidence_factor = predicted_prob_yes - predicted_prob_no

        #  Using a sigmoid activation function in the final layer, returning probabilities instead of direct class predictions.
        #  So from my research I learnt thet a yes values in sigmoid would have to start from a 0.5. But I made mine 0.4 this was due to the fact that our model was not strong enough.
        threshold = 0.41
        prediction_result = "No" if predicted_value <= threshold else "Yes"
        st.success(f"Would this Customer Churn: {prediction_result}. Customers' confidence factor is {confidence_factor[0][0]:.2f}")

# calling the main function to display the result    
if __name__ == '__main__':
  main()
    