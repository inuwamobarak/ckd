import streamlit as st
import joblib
import numpy as np
import keras
# Load the saved model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Make predictions using the loaded model
def predict(model, input_data):
    try:
        # Ensure input_data is a numpy array
        input_data = np.array(input_data)
        
        # Perform predictions
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("CKD Prediction App")
    st.sidebar.header("User Input")
    
    # Define the path to the saved model artifact
    model_path = 'model_CKD.pkl'
    
    # Load the model
    loaded_model = load_model(model_path)
    
    if loaded_model:
        selected_features = [
            'specific_gravity', 'peda_edema', 'red_blood_cell_count', 'appetite', 
            'haemoglobin', 'albumin', 'packed_cell_volume', 'diabetes_mellitus', 
            'blood_glucose_random', 'hypertension'
        ]
        
        input_data = []
        for feature in selected_features:
            feature_value = st.sidebar.number_input(f"Enter {feature}:", step=0.1)
            input_data.append(feature_value)
        
        if st.sidebar.button("Predict"):
            # Make predictions using the loaded model
            predictions = predict(loaded_model, [input_data])
            
            if predictions:
                st.subheader("Prediction:")
                if predictions[0] == 0:
                    st.write("No Chronic Kidney Disease")
                else:
                    st.write("Chronic Kidney Disease")
    
if __name__ == "__main__":
    main()

