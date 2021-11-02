import streamlit as st
import pandas as pd
import numpy


def proccess_dataset(dataset):
    # Read datasend with pandas
    dataset = pd.read_csv("dataset.csv")

    #Output collumns
    for col in dataset.columns:
        st.write(col)

    # Dropping time column since its irrelevant with prediction
    # dataset.drop('time', inplace=True, axis=1)
    #
    # # Making age an int since there are some rows that age is a float
    # dataset['age'] = dataset['age'].astype(numpy.int64)
    # print(dataset)
    # st.write(dataset)


def main():
    # initializing streamlit
    st.title("CEI 471 Semester Project")
    st.header("Group 2")
    st.write("""
            - Christos Christodoulou
            - Danny Qahtan
            - Dimitris Ioannou
            """)

    st.header("Assignment")
    st.write("From historical data given follow the necessary steps to predict the chance of a heart attack and death.")
    dataset = st.file_uploader(label="Select a dataset for the model to train on it.", type=["csv"])
    if dataset is not None:
        proccess_dataset(dataset)


if __name__ == "__main__":
    main()
