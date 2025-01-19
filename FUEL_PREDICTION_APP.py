import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load the saved scaler and model using pickle
with open("fuel_prediction_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


model = tf.keras.models.load_model("fuel_efficiency_prediction_model.h5")
model = tf.keras.models.load_model("fuel_efficiency_prediction_model")


# Streamlit app
def main():
    st.title("Fuel Efficiency Prediction App")
    st.write(
        "Predict the fuel efficiency (MPG) of a vehicle based on its characteristics."
    )

    # Input fields for the features
    cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4, step=1)
    displacement = st.number_input(
        "Displacement (in cubic inches)",
        min_value=50.0,
        max_value=500.0,
        value=150.0,
        step=0.1,
    )
    horsepower = st.number_input(
        "Horsepower", min_value=50.0, max_value=500.0, value=100.0, step=0.1
    )
    weight = st.number_input(
        "Weight (in lbs)", min_value=1000, max_value=6000, value=3000, step=1
    )
    acceleration = st.number_input(
        "Acceleration (0-60 mph in seconds)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.1,
    )
    model_year = st.number_input(
        "Model Year", min_value=1, max_value=100, value=23, step=1
    )

    origin = st.selectbox(
        "Origin",
        options=[1, 2, 3],
        format_func=lambda x: "USA" if x == 1 else "Europe" if x == 2 else "Asia",
    )
    car_code = st.number_input("Car Code", min_value=0, max_value=304, value=0, step=1)
    # Predict button
    if st.button("Predict MPG"):
        try:
            # Combine inputs into a numpy array
            input_data = np.array(
                [
                    [
                        cylinders,
                        displacement,
                        horsepower,
                        weight,
                        acceleration,
                        model_year,
                        origin,
                        car_code,
                    ]
                ]
            )

            # Scale the input data
            scaled_data = scaler.transform(input_data)

            # Predict MPG
            prediction = model.predict(scaled_data)
            st.success(f"Predicted Fuel Efficiency (MPG): {prediction[0][0]:.2f}")
        except Exception as e:
            st.error(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
