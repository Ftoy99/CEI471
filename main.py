import streamlit as st
import pandas as pd
import numpy

# Read datasend with pandas
dataset = pd.read_csv("dataset.csv")
st.title("CEI 471 Semester Project")

# Dropping time column since its irrelevant with prediction
dataset.drop('time', inplace=True, axis=1)

# Making age an int since there are some rows that age is a float
dataset['age'] = dataset['age'].astype(numpy.int64)

print(dataset)
st.write(dataset)
