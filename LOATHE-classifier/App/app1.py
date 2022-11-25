# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Track Utils
# from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table
# Utils
import joblib 
pipe_lr = joblib.load(open("model/pickle_file_final.pkl","rb"))



def main():
	st.title("LOATHE-DETECTION APP")
