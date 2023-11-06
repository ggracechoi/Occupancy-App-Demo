import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image


st.write("""
# Study Spots
## By: Longhorn Learners
""")

#image = Image.open('basketball.png')
#st.image(image, width = 300)

def user_input_features():
    Type = st.sidebar.slider('1 - Quiet, 2 - Average, 3 - Loud', 1, 1, 3)
    Time_to_study = st.sidebar.slider('1 - Chill, 2 - Average, 3 - Grind', 1, 1, 3)
    Lighting = st.sidebar.slider('1 - Low, 2 - High, 3 - Natural', 1, 1, 3)
    data = {'Type': Type,
            'Time_to_study': Time_to_study,
            'Lighting': Lighting,}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('User Parameters')
st.write(df)

data = pd.read_csv('studyspots_data.csv')
X_train = pd.read_csv('X_study_train.csv')
y_train = pd.read_csv('y_study_train.csv')


label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)

if st.button('Submit'):
    model = RandomForestClassifier()
    model = model.fit(X_train, y_train_encoded)
    predictions = model.predict(df)

    st.subheader('Prediction')
    st.write(data.Location[predictions])
