import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score 

st.set_page_config(page_title=" 🚢 Titanic Classifier", layout="wide")
st.title(' 🚢 Titanic Classifier - Предсказание выживаемости пассажиров Титаника')
st.write("## Работа с датасетом титаника")

df = pd.read_csv('titanic_final_processed.csv')
df
