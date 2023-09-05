import streamlit as st
import tensorflow.keras
from keras.models import load_model
model1 = load_model('PulsarClassification.h5')

# Define your app
def app():
    st.title('Pulsar Classification')
    input5 = st.number_input('Mean DMSNR Curve')
    input6 = st.number_input('SD of DMNSR Curve')
    input7 = st.number_input('EK of DMSNR Curve')
    input8 = st.number_input('Skewness of DMSNR Curve')

    # Make a prediction using your model
    prediction = model1.predict([[input5, input6, input7, input8]])

    # Return the prediction as a string
    st.write('Prediction:', 'The Star is classified as a Pulsar' if prediction[0] >= 0.5 else 'The Star is not classified as a Pulsar')

# Run your app
if __name__ == '__main__':
    app()