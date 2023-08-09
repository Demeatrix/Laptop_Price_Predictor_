import streamlit as st
import pickle
import pandas as pd
import numpy as np
from numpy import ndarray
import sklearn

pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pd.read_pickle("df.pkl")

st.title("Laptop Predictor ")

brand = st.selectbox("Choose the Brand of Laptop ", data['Company'].unique())

type_of_lap = st.selectbox('Type of Laptop ', data['TypeName'].unique())

ram_size = st.selectbox('Total RAM (in GB) ', {2, 4, 6, 8, 12, 16, 24, 32, 64})

weight = st.number_input('Weight of the Laptop')

touchscreen = st.selectbox('Touchscreen ', ['Yes', 'No'])

ips = st.selectbox('IPS Display ', ['Yes', 'No'])

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Select Resolution', ['1920x1080', '1366x768', '1600x900', '3840x21060', '3200x1800',
                                                "2880x1080", '2560x1600', '2560x1440', '2304x1440'])

cpu_brand = st.selectbox("Choose the Brand of CPU ", data['CPU Brand'].unique())

hard_drive = st.selectbox('HDD', [0, 128, 256, 512, 1024, 2048])

solid_drive = st.selectbox('SSD', [0, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', data['GPU'].unique())

os = st.selectbox('OS', data['OS'].unique())

if st.button('Predict Price of the Laptop'):
    ppi = None
    if touchscreen == 'YES':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'YES':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([brand, type_of_lap, ram_size, weight, touchscreen, ips, ppi, cpu_brand, hard_drive, solid_drive, gpu, os])

    query = query.reshape(1, 12)
    st.title("The Predicted Price for the chosen Configuration is estimated at: ₹" + str(int(np.exp(pipe.predict(query)[0]))))
