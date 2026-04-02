import streamlit as st
import joblib

# Load model
model = joblib.load("iris_knn_model.pkl")

st.title("🌸 Iris Flower Classification App")

st.write("Enter the flower measurements:")

# Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Predict button
if st.button("Predict"):
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    species = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Species: {result[0]}")