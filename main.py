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
    st.write('text')

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

    col1, col2 = st.columns(2)
    # SEX PLOT
    with col1:
        fig, ax = pyplot.subplots()
        N, bins, patches = ax.hist(dataset["sex"], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7)
        pyplot.xticks((0, 1), ["Male", "Female"])
        pyplot.ylabel("People")
        # ax.set_xticklabels("Male", "Female")
        pyplot.title("Genders")
        patches[0].set_facecolor('royalblue')
        patches[1].set_facecolor('crimson')
        ax.hist(dataset.sex[dataset['DEATH_EVENT'] == 1], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7,
                color="darkorange")
        # Legend
        crimson_patch = mpatches.Patch(color='crimson', label='Women')
        royalblue_patch = mpatches.Patch(color='royalblue', label='Men')
        darkorange_patch = mpatches.Patch(color='darkorange', label='Death')
        pyplot.legend(handles=[crimson_patch, royalblue_patch, darkorange_patch])

        # Data on graph.
        men = len(dataset[(dataset['sex'] == 0)])
        women = len(dataset[(dataset['sex'] == 1)])
        menDied = len(dataset[(dataset['sex'] == 0) & (dataset['DEATH_EVENT'] == 1)])
        womenDied = len(dataset[(dataset['sex'] == 1) & (dataset['DEATH_EVENT'] == 1)])
        pyplot.text(0, men + 1, men)
        pyplot.text(0, menDied + 1, str(round((menDied / men) * 100)) + '%')
        pyplot.text(1, women + 1, women)
        pyplot.text(1, womenDied + 1, str(round((womenDied / women) * 100)) + '%')

        st.pyplot(fig)
    # ANEMIA PLOT
    with col2:
        fig, ax = pyplot.subplots()
        N, bins, patches = ax.hist(dataset["anaemia"], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7)
        pyplot.xticks((0, 1), ["No Anaemia", "Anaemia"])
        pyplot.ylabel("People")
        # ax.set_xticklabels("Male", "Female")
        pyplot.title("Anaemia")
        patches[0].set_facecolor('royalblue')
        patches[1].set_facecolor('crimson')
        ax.hist(dataset.anaemia[dataset['DEATH_EVENT'] == 1], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7,
                color="darkorange")

        # Legend
        crimson_patch = mpatches.Patch(color='crimson', label='Anaemia')
        royalblue_patch = mpatches.Patch(color='royalblue', label='NO Anaemia')
        darkorange_patch = mpatches.Patch(color='darkorange', label='Death')
        pyplot.legend(handles=[crimson_patch, royalblue_patch, darkorange_patch])

        noanemia = len(dataset[(dataset['anaemia'] == 0)])
        anaemia = len(dataset[(dataset['anaemia'] == 1)])
        noanemiaDied = len(dataset[(dataset['anaemia'] == 0) & (dataset['DEATH_EVENT'] == 1)])
        anemiaDied = len(dataset[(dataset['anaemia'] == 1) & (dataset['DEATH_EVENT'] == 1)])
        pyplot.text(0, noanemia + 1, noanemia)
        pyplot.text(0, noanemiaDied + 1, str(round((noanemiaDied / noanemia) * 100)) + '%')
        pyplot.text(1, anaemia + 1, anaemia)
        pyplot.text(1, anemiaDied + 1, str(round((anemiaDied / anaemia) * 100)) + '%')

        st.pyplot(fig)

    col3, col4 = st.columns(2)
    # SEX PLOT
    with col3:
        fig, ax = pyplot.subplots()
        # bins ???
        N, bins, patches = ax.hist(dataset["diabetes"], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7)
        pyplot.xticks((0, 1), ["No Diabetes", "Diabetes"])
        pyplot.ylabel("People")
        # ax.set_xticklabels("Male", "Female")
        pyplot.title("Diabetes")
        patches[0].set_facecolor('royalblue')
        patches[1].set_facecolor('crimson')
        ax.hist(dataset.diabetes[dataset['DEATH_EVENT'] == 1], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7,
                color="darkorange")
        # Legend
        crimson_patch = mpatches.Patch(color='crimson', label='Diabetes')
        royalblue_patch = mpatches.Patch(color='royalblue', label='No Diabetes')
        darkorange_patch = mpatches.Patch(color='darkorange', label='Death')
        pyplot.legend(handles=[crimson_patch, royalblue_patch, darkorange_patch])

        nodiabetes = len(dataset[(dataset['diabetes'] == 0)])
        diabetes = len(dataset[(dataset['diabetes'] == 1)])
        nodiabetesDied = len(dataset[(dataset['diabetes'] == 0) & (dataset['DEATH_EVENT'] == 1)])
        diabetesDied = len(dataset[(dataset['diabetes'] == 1) & (dataset['DEATH_EVENT'] == 1)])
        pyplot.text(0, nodiabetes + 1, nodiabetes)
        pyplot.text(0, nodiabetesDied + 1, str(round((nodiabetesDied / nodiabetes) * 100)) + '%')
        pyplot.text(1, diabetes + 1, diabetes)
        pyplot.text(1, diabetesDied + 1, str(round((diabetesDied / diabetes) * 100)) + '%')

        st.pyplot(fig)
    # Smoking PLOT
    with col4:
        fig, ax = pyplot.subplots()
        N, bins, patches = ax.hist(dataset["smoking"], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7)
        pyplot.xticks((0, 1), ["No Smoking", "Smoking"])
        pyplot.ylabel("People")
        # ax.set_xticklabels("Male", "Female")
        pyplot.title("Smoking")
        patches[0].set_facecolor('royalblue')
        patches[1].set_facecolor('crimson')
        ax.hist(dataset.smoking[dataset['DEATH_EVENT'] == 1], bins=[-.5, .5, 1.5], ec="dimgray", rwidth=0.7,
                color="darkorange")

        # Legend
        crimson_patch = mpatches.Patch(color='crimson', label='Smoking')
        royalblue_patch = mpatches.Patch(color='royalblue', label='No Smoking')
        darkorange_patch = mpatches.Patch(color='darkorange', label='Death')
        pyplot.legend(handles=[crimson_patch, royalblue_patch, darkorange_patch])

        nosmoking = len(dataset[(dataset['smoking'] == 0)])
        smoking = len(dataset[(dataset['smoking'] == 1)])
        nosmokingDied = len(dataset[(dataset['smoking'] == 0) & (dataset['DEATH_EVENT'] == 1)])
        smokingDied = len(dataset[(dataset['smoking'] == 1) & (dataset['DEATH_EVENT'] == 1)])
        pyplot.text(0, nosmoking + 1, nosmoking)
        pyplot.text(0, nosmokingDied + 1, str(round((nosmokingDied / nosmoking) * 100)) + '%')
        pyplot.text(1, smoking + 1, smoking)
        pyplot.text(1, smokingDied + 1, str(round((smokingDied / smoking) * 100)) + '%')

        rcParams['figure.figsize'] = 20, 14
        pyplot.matshow(dataset.corr())
        pyplot.yticks(np.arange(dataset.shape[1]), dataset.columns)
        pyplot.xticks(np.arange(dataset.shape[1]), dataset.columns)
        pyplot.colorbar()

        st.pyplot(fig)

    col5, col6 = st.columns(2)
    with col5:
        fig = sns.catplot(x="DEATH_EVENT", y="age", hue="smoking", kind="bar", data=dataset)

        pyplot.title('Death Incidents from High Blood Pressure', size=25)
        pyplot.xlabel('Death People', size=20)
        pyplot.ylabel('People', size=20)

        st.pyplot(fig)

    with col6:
        fig, ax = pyplot.subplots()

        x = [0, 1]
        default_x_ticks = range(len(x))
        pyplot.xticks(default_x_ticks, x, size=30)
        pyplot.yticks(size=25)
        pyplot.margins(0.5, 0.1)

        pyplot.title("Test", size=75)
        pyplot.xlabel("High Blood Pressure", size=50)
        pyplot.ylabel("Age", size=50)
        x = dataset.high_blood_pressure[dataset['DEATH_EVENT'] == 0]
        y = dataset.age[dataset['DEATH_EVENT'] == 0]
        pyplot.scatter(x, y, c="darkorange")

        k = dataset.sex[dataset['DEATH_EVENT'] == 1]
        l = dataset.age[dataset['DEATH_EVENT'] == 1]
        pyplot.scatter(k, l, c="dimgray")
        pyplot.legend(["No Death", "Death"])
        st.pyplot(fig)


def machine_learning(dataset, model, testPercentage):
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
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Models go here
    if model == 'Logistic Regression':
        mlModel = LogisticRegression(random_state=0)
    elif model == 'K-Nearest Neighbors':
        mlModel = KNeighborsClassifier()
    elif model == 'Support Vector Machine':
        mlModel = SVC(random_state=42,probability=True)
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
            myPredictionData = pd.DataFrame(data)
            st.write(myPredictionData)
            # TODO Fix warning here
            print(data)
            st.write(mlModel.predict(data))
            st.write(mlModel.predict_proba(data))
            print(mlModel.predict(data))
            print(mlModel.predict_proba(data))


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
