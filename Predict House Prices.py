import streamlit as st
import pickle
import numpy as np

model1 = pickle.load(open("data/linear_reg.model", 'rb'))

st.title("KNOW THE PRICE OF YOUR PREFERRED HOME!")

st.sidebar.write("## Predict House Prices")
st.write("### Just input/pick your preferences below")

# area,bedrooms,bathrooms,stories,
# mainroad,guestroom,basement,hotwaterheating,
# airconditioning,parking,prefarea,furnishingstatus

area = st.slider("Area in sq metres", 0, 15000)
mainroad = st.radio("A house along the main road?", ["Yes", "No"])
bedrooms = st.slider("Number of bedrooms", 0, 10)
guestroom = st.radio("Hav a guest room?", ["Yes", "No"])
bathrooms = st.slider("Number of bathrooms", 0, 10)
basement = st.radio("Have a basement?", ["Yes", "No"])
stories = st.slider("Number of stories", 0, 5)
hotwaterheating = st.radio("Have hot water heating?", ["Yes", "No"])
parking = st.slider("Parking spaces", 0, 5)
airconditioning = st.radio("Have air conditioning?", ["Yes", "No"])
prefarea = st.radio("Playing ground?", ["Yes", "No"])
furnishingstatus = st.radio("Level of furnishing", ["Furnished", "Semi-furnished", "Not furnished"])


def to_integers(response):
    if response == "Yes" or response == "Furnished":
        return 1
    elif response == "No" or response == "Not furnished":
        return 0
    elif response == "Semi-furnished":
        return 2


def input_data(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus):
    mainroad = to_integers(mainroad)
    guestroom = to_integers(guestroom)
    basement = to_integers(basement)
    hotwaterheating = to_integers(hotwaterheating)
    airconditioning = to_integers(airconditioning)
    prefarea = to_integers(prefarea)
    furnishingstatus = to_integers(furnishingstatus)
    input_array = [area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning,
                   parking, prefarea, furnishingstatus]

    return input_array


def predict(data):
    result = model1.predict(data)
    return result


data = input_data(mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea, furnishingstatus)
data = np.array(data)
data = data.reshape(-1, 12)
price = predict(data=data)
st.sidebar.write("## The Price of your House would be Ksh.{}".format(int(price)))
