import streamlit as st
import pandas as pd
import numpy
import matplotlib.pyplot as pyplot
import matplotlib
import matplotlib.colors as colors
from matplotlib import *
import matplotlib.patches as mpatches


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
    del (dataset["time"])
    # Making age an int since there are some rows that age is a float
    st.write(dataset)
    st.write("Our dataset looks clean enough so that it can be processed.")
    return dataset


def data_analysis(dataset):
    st.header('Data Analysis and Statistics')
    st.write('text')
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = pyplot.subplots()
        # bins ???
        N, bins, patches = ax.hist(dataset["sex"], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7)
        pyplot.xticks((0, 1), ["Male", "Female"])
        pyplot.ylabel("People")
        # ax.set_xticklabels("Male", "Female")
        pyplot.title("Genders")
        patches[0].set_facecolor('royalblue')
        patches[1].set_facecolor('crimson')
        ax.hist(dataset.sex[dataset['DEATH_EVENT'] == 1], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7, color="darkorange")
        #Legend
        crimson_patch = mpatches.Patch(color='crimson', label='Women')
        royalblue_patch = mpatches.Patch(color='royalblue', label='Men')
        darkorange_patch = mpatches.Patch(color='darkorange', label='Death')
        pyplot.legend(handles=[crimson_patch,royalblue_patch,darkorange_patch])
        st.pyplot(fig)
    with col2:
        fig, ax = pyplot.subplots()
        # bins ???
        N, bins, patches = ax.hist(dataset["sex"], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7)
        pyplot.xticks((0, 1), ["Male", "Female"])
        pyplot.ylabel("People")
        # ax.set_xticklabels("Male", "Female")
        pyplot.title("Genders")
        patches[0].set_facecolor('royalblue')
        patches[1].set_facecolor('crimson')
        ax.hist(dataset.sex[dataset['DEATH_EVENT'] == 1], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7, color="red")
        st.pyplot(fig)


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
        data_analysis(dataset)


if __name__ == "__main__":
    main()
