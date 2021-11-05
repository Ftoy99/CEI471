import streamlit as st
import pandas as pd
import numpy


def clean_dataset(dataset):
    st.header('Data Cleaning')
    st.write('We are going to process our data to better suit our needs.')
    # TODO MAKE WRITE BEAUTIFUL 1
    st.write(dataset)
    st.write(
        "Age and platelets are integers so we are going to cast them accordingly. We are also going to set the precision to 2 decimals. Also "
        "time is not relevant with predicting the chance of heart attack so it can be dropped.")
    code = '''    dataset['age'] = dataset['age'].astype(numpy.int64)
    dataset['platelets'] = dataset['platelets'].astype(numpy.int64)
    dataset['serum_creatinine'] = dataset['serum_creatinine'].map('{:,.2f}'.format)
    del(dataset["time"])
    '''
    st.code(code, language='python')
    dataset['age'] = dataset['age'].astype(numpy.int64)
    dataset['platelets'] = dataset['platelets'].astype(numpy.int64)
    dataset['serum_creatinine'] = dataset['serum_creatinine'].map('{:,.2f}'.format)
    del(dataset["time"])
    # Making age an int since there are some rows that age is a float
    st.write(dataset)
    st.write("Our dataset looks clean enough so that it can be processed.")
    return dataset


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
        dataset = clean_dataset(dataset)
    st.header("Graphs And Statistics of Dataset")


if __name__ == "__main__":
    main()
