import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib
import matplotlib.colors as colors
from matplotlib import *
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def pred_heart_attack(data, model):
    print(model.predict(data))
    print(model.predict_proba(data))


def clean_dataset(dataset):
    st.header('Data Cleaning')
    st.write('We are going to process our data to better suit our needs.')
    # TODO MAKE WRITE BEAUTIFUL 1
    st.write(dataset)
    st.write(
        "Age and platelets are integers so we are going to cast them accordingly. We are also going to set the "
        "precision to 2 decimals. Also "
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
    # Clean dataset
    dataset['age'] = dataset['age'].astype(numpy.int64)
    dataset['platelets'] = dataset['platelets'].astype(numpy.int64)
    dataset['serum_creatinine'] = dataset['serum_creatinine'].map('{:,.2f}'.format)
    del (dataset["time"])
    st.header('Data Analysis and Statistics')

    with st.expander("Correlation Matrix"):
        st.write("""
           Correlation Matrix- let’s you see correlations between all variables.

    Within seconds, you can see whether something is positively or negatively correlated with our predictor (Death event).
    
        """)

        code = '''# Corellation Matrix
    corr_matrix = dataset.corr()
    fig, ax = pyplot.subplots()
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidths=0.5,
                     fmt=".2f",
                     cmap="YlGnBu")
                  '''
        st.code(code, language='python')




    # Corellation Matrix
    corr_matrix = dataset.corr()
    fig, ax = pyplot.subplots()
    ax = sns.heatmap(corr_matrix,
                     annot=True,
                     linewidths=0.5,
                     fmt=".2f",
                     cmap="YlGnBu")

    bottom, top = ax.get_ylim()
    st.pyplot(fig)





    # #PairPlot
    # subData = dataset[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium']]
    # pyplot.margins(0)
    # fig = sns.pairplot(subData)
    # st.pyplot(fig)

    # COMPARISON PAIRPLOT
    st.header('COMPARISON PLOT')

    with st.expander("Comparison Plot Code"):
        st.write("""
             

           """)

        code = '''
        #SET AND DISPLAY FILTERING FOR EACH AXIS AND DEATH CHECKBOX
    sel = st.selectbox('Select Attribute for X Axis',
                       ('age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium'))

    sel2 = st.selectbox('Select Attribute for Y Axis',
                        ('age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium'))

    death = st.checkbox('Show death?', key=1)
    
    #IF DEATH IS CHECKED SET AND DISPLAY TARGET ON THE PLOT
    
        if death:

        if sel == 'age':
            x1 = dataset.age[dataset['DEATH_EVENT'] == 0]
            x2 = dataset.age[dataset['DEATH_EVENT'] == 1]
        elif sel == 'creatinine_phosphokinase':
            x1 = dataset.creatinine_phosphokinase[dataset['DEATH_EVENT'] == 0]
            x2 = dataset.creatinine_phosphokinase[dataset['DEATH_EVENT'] == 1]
            
            ... #for each data element
    
    
    #SET THE PLOT
    
    pyplot.scatter(x1, y1, c="darkorange")
        pyplot.scatter(x2, y2, c="dimgray")
        pyplot.xlabel(sel, size=40)
        pyplot.ylabel(sel2, size=40)
        pyplot.legend(["No Death", "Death"])
        st.pyplot(fig)

                     '''
        st.code(code, language='python')

    fig, ax = pyplot.subplots()

    sel = st.selectbox('Select Attribute for X Axis',
                       ('age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium'))

    sel2 = st.selectbox('Select Attribute for Y Axis',
                        ('age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_sodium'))

    death = st.checkbox('Show death?', key=1)

    if death:

        if sel == 'age':
            x1 = dataset.age[dataset['DEATH_EVENT'] == 0]
            x2 = dataset.age[dataset['DEATH_EVENT'] == 1]
        elif sel == 'creatinine_phosphokinase':
            x1 = dataset.creatinine_phosphokinase[dataset['DEATH_EVENT'] == 0]
            x2 = dataset.creatinine_phosphokinase[dataset['DEATH_EVENT'] == 1]
        elif sel == 'ejection_fraction':
            x1 = dataset.ejection_fraction[dataset['DEATH_EVENT'] == 0]
            x2 = dataset.ejection_fraction[dataset['DEATH_EVENT'] == 1]
        elif sel == 'platelets':
            x1 = dataset.platelets[dataset['DEATH_EVENT'] == 0]
            x2 = dataset.platelets[dataset['DEATH_EVENT'] == 1]
        elif sel == 'serum_sodium':
            x1 = dataset.serum_sodium[dataset['DEATH_EVENT'] == 0]
            x2 = dataset.serum_sodium[dataset['DEATH_EVENT'] == 1]

        if sel2 == 'age':
            y1 = dataset.age[dataset['DEATH_EVENT'] == 0]
            y2 = dataset.age[dataset['DEATH_EVENT'] == 1]
        elif sel2 == 'creatinine_phosphokinase':
            y1 = dataset.creatinine_phosphokinase[dataset['DEATH_EVENT'] == 0]
            y2 = dataset.creatinine_phosphokinase[dataset['DEATH_EVENT'] == 1]
        elif sel2 == 'ejection_fraction':
            y1 = dataset.ejection_fraction[dataset['DEATH_EVENT'] == 0]
            y2 = dataset.ejection_fraction[dataset['DEATH_EVENT'] == 1]
        elif sel2 == 'platelets':
            y1 = dataset.platelets[dataset['DEATH_EVENT'] == 0]
            y2 = dataset.platelets[dataset['DEATH_EVENT'] == 1]
        elif sel2 == 'serum_sodium':
            y1 = dataset.serum_sodium[dataset['DEATH_EVENT'] == 0]
            y2 = dataset.serum_sodium[dataset['DEATH_EVENT'] == 1]

        pyplot.scatter(x1, y1, c="darkorange")
        pyplot.scatter(x2, y2, c="dimgray")
        pyplot.xlabel(sel, size=40)
        pyplot.ylabel(sel2, size=40)
        pyplot.legend(["No Death", "Death"])
        st.pyplot(fig)

    else:
        if sel == 'age':
            x = dataset.age
        elif sel == 'creatinine_phosphokinase':
            x = dataset.creatinine_phosphokinase
        elif sel == 'ejection_fraction':
            x = dataset.ejection_fraction
        elif sel == 'platelets':
            x = dataset.platelets
        elif sel == 'serum_sodium':
            x = dataset.serum_sodium

        if sel2 == 'age':
            y = dataset.age
        elif sel2 == 'creatinine_phosphokinase':
            y = dataset.creatinine_phosphokinase
        elif sel2 == 'ejection_fraction':
            y = dataset.ejection_fraction
        elif sel2 == 'platelets':
            y = dataset.platelets
        elif sel2 == 'serum_sodium':
            y = dataset.serum_sodium;

        pyplot.scatter(x, y, c="darkorange")
        pyplot.xlabel(sel, size=25)
        pyplot.ylabel(sel2, size=25)
        st.pyplot(fig)

    # col1, col2 = st.columns(2)
    # # SEX PLOT
    #
    # with col1:
    #     fig, ax = pyplot.subplots()
    #     N, bins, patches = ax.hist(dataset["sex"], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7)
    #     pyplot.xticks((0, 1), ["Male", "Female"])
    #     pyplot.ylabel("People")
    #     # ax.set_xticklabels("Male", "Female")
    #     pyplot.title("Genders")
    #     patches[0].set_facecolor('royalblue')
    #     patches[1].set_facecolor('crimson')
    #     ax.hist(dataset.sex[dataset['DEATH_EVENT'] == 1], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7,
    #             color="darkorange")
    #     # Legend
    #     crimson_patch = mpatches.Patch(color='crimson', label='Women')
    #     royalblue_patch = mpatches.Patch(color='royalblue', label='Men')
    #     darkorange_patch = mpatches.Patch(color='darkorange', label='Death')
    #     pyplot.legend(handles=[crimson_patch, royalblue_patch, darkorange_patch])
    #
    #     # Data on graph.
    #     men = len(dataset[(dataset['sex'] == 0)])
    #     women = len(dataset[(dataset['sex'] == 1)])
    #     menDied = len(dataset[(dataset['sex'] == 0) & (dataset['DEATH_EVENT'] == 1)])
    #     womenDied = len(dataset[(dataset['sex'] == 1) & (dataset['DEATH_EVENT'] == 1)])
    #     pyplot.text(0, men + 1, men)
    #     pyplot.text(0, menDied + 1, str(round((menDied / men) * 100)) + '%')
    #     pyplot.text(1, women + 1, women)
    #     pyplot.text(1, womenDied + 1, str(round((womenDied / women) * 100)) + '%')
    #
    #     st.pyplot(fig)

    st.header('COUNT PLOT')

    with st.expander("Count Plot Code"):
        st.write("""


           """)

        code = '''
        
         #SET AND DISPLAY FILTERING FOR EACH AXIS AND DEATH CHECKBOX
         
        sel3 = st.selectbox('Select Attribute',
                        ('anaemia', 'diabetes', 'high_blood_pressure', 'smoking'))

    death2 = st.checkbox('Show death?', key=2)
    
          #IF DEATH IS CHECKED SET AND DISPLAY TARGET ON THE PLOT

    if death2:
        fig = sns.catplot(x=sel3, hue="DEATH_EVENT", kind="count", data=dataset)

        pyplot.title('Death Incidents from ' + sel3, size=25)
        pyplot.xticks((0, 1), ["No " + sel3, sel3])
        pyplot.xlabel(sel3, size=20)
        pyplot.ylabel('People', size=20)
        st.pyplot(fig)
    else:
        fig = sns.catplot(x=sel3, kind="count", data=dataset)
        pyplot.xticks((0, 1), ["No " + sel3, sel3])
        pyplot.xlabel(sel3, size=20)
        pyplot.ylabel('People', size=20)
        st.pyplot(fig)

                     '''
        st.code(code, language='python')

    sel3 = st.selectbox('Select Attribute',
                        ('anaemia', 'diabetes', 'high_blood_pressure', 'smoking'))

    death2 = st.checkbox('Show death?', key=2)

    if death2:
        fig = sns.catplot(x=sel3, hue="DEATH_EVENT", kind="count", data=dataset)

        pyplot.title('Death Incidents from ' + sel3, size=25)
        pyplot.xticks((0, 1), ["No " + sel3, sel3])
        pyplot.xlabel(sel3, size=20)
        pyplot.ylabel('People', size=20)
        st.pyplot(fig)
    else:
        fig = sns.catplot(x=sel3, kind="count", data=dataset)
        pyplot.xticks((0, 1), ["No " + sel3, sel3])
        pyplot.xlabel(sel3, size=20)
        pyplot.ylabel('People', size=20)
        st.pyplot(fig)


def machine_learning(dataset, model, testPercentage):
    with st.expander("Machine Learning"):
        st.write(
            "The machine Learning part of this streamlit programs initializes with the model selected from the sidebar.")
        code = '''
           if model == 'Logistic Regression':
               mlModel = LogisticRegression(random_state=0)
           elif model == 'K-Nearest Neighbors':
               mlModel = KNeighborsClassifier()
           elif model == 'Support Vector Machine':
               mlModel = SVC(random_state=42, probability=True)
           '''

        st.code(code, language='python')
        st.write("Next we split our dataset into training and testing data. We use the percentage from the sidebar.")
        code = '''X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=int(testPercentage) / 100,random_state=42)
           '''
        st.code(code, language='python')

        st.write("Now with our split data we train our model.")
        code = '''mlModel.fit(X_train, Y_train)
           '''
        st.code(code, language='python')

        st.write("To make predictions we use the predict function with a dataframe."
                 "Note:The method returns a dataframe collum with the expected output.")
        code = '''mlModel.predict(myDataFrame)'''
        st.code(code, language='python')
        st.write("To get a probability output we must use the probability method.")
        code = '''mlModel.predict_proba(myDataFrame)'''
        st.code(code, language='python')

    # Clean
    dataset['age'] = dataset['age'].astype(numpy.int64)
    dataset['platelets'] = dataset['platelets'].astype(numpy.int64)
    dataset['serum_creatinine'] = dataset['serum_creatinine'].map('{:,.2f}'.format)
    del (dataset["time"])
    # Split dataset .
    X = dataset.drop('DEATH_EVENT', axis=1)
    Y = dataset['DEATH_EVENT']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=int(testPercentage) / 100,
                                                        random_state=42)
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    # Models go here
    if model == 'Logistic Regression':
        mlModel = LogisticRegression(random_state=0)
    elif model == 'K-Nearest Neighbors':
        mlModel = KNeighborsClassifier()
    elif model == 'Support Vector Machine':
        mlModel = SVC(random_state=42, probability=True)
    mlModel.fit(X_train, Y_train)
    pred = mlModel.predict(X_train)

    st.sidebar.write(f"Accuracy Score on trained data: {accuracy_score(Y_train, pred) * 100:.2f}%")

    pred = mlModel.predict(X_test)
    st.sidebar.write(f"Accuracy Score on untrained data: {accuracy_score(Y_test, pred) * 100:.2f}%")

    with st.form("predict_heart_attack_form"):
        st.write('Predict Heart Attack')
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input('Age', step=1, min_value=0, max_value=100)
            anemia = st.selectbox('Anemia', ('Yes', 'No'))
            creatine = st.number_input('Creatinine Phosphokinase')
            ejectionFraction = st.number_input(label='Ejection_Fraction', step=1, min_value=0, max_value=100)

        with col2:
            sex = st.selectbox('Gender', ('Man', 'Woman'))
            diabetes = st.selectbox('Diabetes', ('Yes', 'No'))
            platelets = st.number_input(label='Platelets', step=1000, min_value=0)
            serumSodium = st.number_input(label='Serum Sodium', step=1, min_value=0)

        with col3:
            smoking = st.selectbox('Smoking', ('Yes', 'No'))
            highBP = st.selectbox('High blood Pressure', ('Yes', 'No'))
            serumCreatinine = st.number_input(label='Serum Creatinine', step=0.01, min_value=0.00)

        submitted = st.form_submit_button("Predict")
        # Process values from form .
        if submitted:
            if anemia == 'Yes':
                anemia = 1
            else:
                anemia = 0

            if sex == 'Woman':
                sex = 1
            else:
                sex = 0

            if smoking == 'Yes':
                smoking = 1
            else:
                smoking = 0

            if highBP == 'Yes':
                highBP = 1
            else:
                highBP = 0

            if diabetes == 'Yes':
                diabetes = 1
            else:
                diabetes = 0
            data = [[int(age), anemia, creatine, diabetes, ejectionFraction, highBP, platelets, serumCreatinine,
                     serumSodium,
                     sex, smoking], ]
            myPredictionData = pd.DataFrame(dat            output = "The chance of the individual having a heart attack is : " + str(int(round(float(mlModel.predict_proba(myPredictionData)[0][1]*100),0)))+"%"
            st.write(output)


def main():
    # initializing streamlit
    st.sidebar.title("CEI 471 Semester Project")
    st.sidebar.write('Heart Attack Prediction With Machine Learning')
    st.sidebar.header("Group 2")
    st.sidebar.write("""
            - Christos Christodoulou
            - Danny Kahtan
            - Dimitris Ioannou
            """)
    dataset = st.sidebar.file_uploader(label="Load your dataset.", type=["csv"])
    # Form for Machine Learning
    if dataset is not None:
        dataset = pd.read_csv(dataset)
        op = st.sidebar.selectbox('Select View',
                                  ('Data Cleaning', 'Data Analysis', 'Machine Learning'))
        if op == 'Data Cleaning':
            clean_dataset(dataset)
        elif op == 'Data Analysis':
            data_analysis(dataset)
        elif op == 'Machine Learning':
            model = st.sidebar.selectbox('Chose Machine Learning Algorithm',
                                         ('Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine'))
            testPercentage = st.sidebar.slider('Datasize percentage for testing.', 10, 90, 20)
            machine_learning(dataset, model, testPercentage)


if __name__ == "__main__":
    main()
