import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import plotly.express as px
import os
pd.options.display.float_format = '{:.2f}'.format

#set title of the page

st.set_page_config(page_title="Segmentation", page_icon=":bank:",layout="wide")

st.title(" :bank: Customer Segmentation Dashboard")

#load dataset
@st.cache_data
def load_data(path: str):
    data = pd.read_csv(path)
    return data

df = load_data("final_transaction.csv")
final_data = df.sample(frac=.1)
df = df.drop('Unnamed: 0', axis=1)
st.write(df)
