import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.set_page_config(page_title=" 🚢 Titanic Classifier", layout="wide")
st.title(' 🚢 Titanic Classifier - Предсказание выживаемости пассажиров Титаника')
st.write("## Работа с датасетом титаника")

@st.cache_data
def load_data():
  df = pd.read_csv('titanic_final_processed.csv')
   return df
@st.cache_resource
def train_model(df):
    st.write("Обучение модели...")

st.subheader("Случайные 10 строк")
st.dataframe(df.sample(10), use_container_width=True)

st.subheader("Визуализация данных")
col1, col2 = st.columns(2)
with col1:
  fig1 = px.histogram(df, x="Survived", color="FamilySize", barmode="group", title="Выживаемость исходя от количество членов семьи")
  st.plotly_chart(fig1, use_container_width=True)
with col2:
  fig2 = px.scatter(df, x="Age", y="Fare", color="Survived", title="Возрасть vs Стоимость билета")
  st.plotly_chart(fig2, use_container_width=True)
X = df.drop(['Survived'], axis=1)
y = df['Survived']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


