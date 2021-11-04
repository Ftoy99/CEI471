import streamlit as st
import pandas as pd
import numpy


def clean_dataset(dataset):
    # Making age an int since there are some rows that age is a float
    # dataset['age'] = dataset['age'].astype(numpy.int64)
    st.header("Data Cleaning")
    st.write("We are going to process our data to better suit our needs.")

    # TODO MAKE WRITE BEAUTIFUL 1
    st.write(dataset)
    st.write("First we are going to drop unnecessary columns we don't need.")
    colToDrop = st.multiselect(label="Select columns to drop.", options=dataset.columns)
    for x in colToDrop:
        dataset.drop(x, inplace=True, axis=1)
    # TODO MAKE WRITE BEAUTIFUL 2
    st.write(dataset)
    # Setting Precision for data.
    st.write("Adjust the precision and remove all unnecessary decimal digits")
    values = {}
    for x in dataset.columns:
        if dataset[x].dtype == "float64":
            values[x] = [st.slider(x, 0, 64, 3), x]


def main():
    # initializing streamlit
    # TODO ADD STUFF IN SIDEBAR ?
    st.sidebar.selectbox("smth", "optins in list here")
    with st.container():
        st.title("CEI 471 Semester Project")
        st.header("Group 2")
        st.write("""
                - Christos Christodoulou
                - Danny Kahtan
                - Dimitris Ioannou
                """)
    with st.container():
        st.header("Assignment")
        st.write(
            "From historical data given follow the necessary steps to predict the chance of a heart attack and death.")
        dataset = st.file_uploader(label="Select a dataset for the model to train on it.", type=["csv"])
    if dataset is not None:
        dataset = pd.read_csv(dataset)
        clean_dataset(dataset)


if __name__ == "__main__":
    main()
