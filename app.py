import streamlit as st
import joblib
import numpy as np

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

# Decode categorical labels for hypertension, diabetes_mellitus, appetite, and peda_edema
def decode_labels(label):
    return "Yes" if label == 1 else "No"

def main():
    # Set the title and description of the app
    st.title("CKD Prediction App")
    st.write("This app predicts Chronic Kidney Disease (CKD) based on input features.")

    # Load the model
    model_path = 'model_CKD.pkl'
    loaded_model = load_model(model_path)

    if loaded_model:
        # Create input fields for user input
        st.subheader("Enter Input Features:")
        
        # Input fields for specific features with default values
        specific_gravity = st.slider("Specific Gravity", 0.0, 1.0, 0.75)
        albumin = st.slider("Albumin", 0, 5, 0)
        blood_glucose_random = st.slider("Blood Glucose (Random)", 0.0, 1.0, 0.211538)
        haemoglobin = st.slider("Haemoglobin", 0.0, 1.0, 0.836735)
        packed_cell_volume = st.slider("Packed Cell Volume", 0.0, 1.0, 0.777778)
        red_blood_cell_count = st.slider("Red Blood Cell Count", 0.0, 1.0, 0.525424)
        hypertension = st.radio("Hypertension", ("No", "Yes"), index=1)  # Decoded labels
        diabetes_mellitus = st.radio("Diabetes Mellitus", ("No", "Yes"), index=0)  # Decoded labels
        appetite = st.radio("Appetite", ("No", "Yes"), index=0)  # Decoded labels
        peda_edema = st.radio("Peda Edema", ("No", "Yes"), index=0)  # Decoded labels
        
        # Create a button to trigger predictions
        if st.button("Predict"):
            # Prepare input data as a list (in the same order as your model's input features)
            input_data = [[
                specific_gravity, albumin, blood_glucose_random, haemoglobin, packed_cell_volume,
                red_blood_cell_count, 1 if hypertension == "Yes" else 0,  # Encode labels back to 1 or 0
                1 if diabetes_mellitus == "Yes" else 0,  # Encode labels back to 1 or 0
                1 if appetite == "Yes" else 0,  # Encode labels back to 1 or 0
                1 if peda_edema == "Yes" else 0  # Encode labels back to 1 or 0
            ]]
            
            # Make predictions
            predictions = predict(loaded_model, input_data)
            
            # Display the predictions
            if predictions is not None:
                st.subheader("Predictions:")
                # You can customize the output format based on your model's output
                st.write(f"Probability of CKD: {predictions[0]:.2f}")
                
if __name__ == "__main__":
    main()

