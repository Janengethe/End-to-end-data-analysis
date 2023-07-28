import streamlit as st
import pandas as pd
import plotly.express as px


st.set_option('deprecation.showPyplotGlobalUse', False)


# Load the dataset

def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

st.title("Titanic Dataset Analysis")
st.write("By: Jane Ng'ethe")


# Display the data
if st.checkbox("Show Data"):
    st.subheader("Raw Data")
    st.dataframe(data)

# Data Preprocessing
# Check for missing values
st.subheader("Missing Values")
st.dataframe(data.isnull().sum())

# Handle missing values
# Fill missing values for age
data["Age"].fillna(data["Age"].mean(), inplace=True)

# Drop cabin column since most of the rows have null values
data.drop("Cabin", axis=1, inplace=True)

# Fill the Embarked with mode
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

st.subheader("Handled Missing Values")
st.dataframe(data.isnull().sum())


# Feature engineering
# Add Family size column to contain the total number of people in that family plus themselves
data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

# Create a new binary feature indicating whether a passenger is traveling alone or with family
data["IsAlone"] = data["FamilySize"].apply(lambda x: 1 if x == 1 else 0)


st.subheader("Feature Engineering")
st.write("Added FamilySize and IsAlone Columns")
st.dataframe(data)


# Data Visualization

st.subheader("Data Visualization")

survival_data = data.groupby("Sex")["Survived"].mean().reset_index()
fig = px.bar(survival_data, x="Sex", y="Survived", color="Sex", title='Survival Rate by Gender')
st.plotly_chart(fig)


survival_data = data.groupby("Pclass")["Survived"].mean().reset_index()
fig = px.bar(survival_data, x="Pclass", y="Survived", title='Survival Rate by Passenger Class', color="Pclass")
st.plotly_chart(fig)


fig = px.histogram(data, x="Age", nbins=20, color="Survived", title='Age Distribution of Passengers')
st.plotly_chart(fig)
