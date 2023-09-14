import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# styling to center the image and title in sidebar
st.markdown(
    """
    <style>
        [data-testid=stSidebar]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

# sidebar for the app
with st.sidebar:
    # image/logo
    st.image("https://cdn-icons-png.flaticon.com/512/2172/2172891.png", width=100)
    # title of the app
    st.title("AutoML App")
    # choices for the app
    choice = st.radio("", ["Upload Data", "Auto-EDA", "Train Models", "Download Best Model"])
    # basic information for the app
    st.info("Upload your dataset to automatically build a machine learning pipeline and download it to use on your local machine!")

if os.path.exists("uploaded_data.csv"):
        df = pd.read_csv("uploaded_data.csv", index_col=None)

def is_data_uploaded():
    if os.path.exists("uploaded_data.csv"):
        return True
    else:
        return False

# for upload option
if choice == "Upload Data":
    if is_data_uploaded():
        os.remove("uploaded_data.csv")
    file = st.file_uploader("Upload Data (CSV only)")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("uploaded_data.csv", index=None)
        st.dataframe(df)

if choice == "Auto-EDA":
    if is_data_uploaded():
        st.title("Automated Data Analysis")
        report = df.profile_report()
        st_profile_report(report)
        st.download_button("Download Report", report.to_html(), file_name="report.html", mime="text/html")


    else:
        st.title("Automated Data Analysis")
        st.info("Make sure you've uploaded the dataset!")
    

if choice == "Train Models":
    if is_data_uploaded():
        st.title("Automated ML Model Trainer")
    else:
        st.title("Automated ML Model Trainer")
        st.info("Make sure you've uploaded the dataset!")

if choice == "Download Best Model":
    if is_data_uploaded():
        st.title("Download Best Model")
    else:
        st.title("Download Best Model")
        st.info("Make sure you've uploaded the dataset and trained ML models!")
