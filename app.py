import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.regression import setup as rsetup, pull as rpull, compare_models as rcompare_models, save_model as rsave_model
from pycaret.classification import setup as csetup, pull as cpull, compare_models as ccompare_models, save_model as csave_model

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

# making the uploaded dataset available for all choices
if os.path.exists("uploaded_data.csv"):
        df = pd.read_csv("uploaded_data.csv", index_col=None)

# function to check whether the dataset has been uploaded or not
def is_data_uploaded():
    if os.path.exists("uploaded_data.csv"):
        return True
    else:
        return False

# function to check whether the model has been trained or not  
def is_model_trained():
    if os.path.exists("best_model.pkl"):
        return True
    else:
        return False

# for upload option
if choice == "Upload Data":
    # if data is uploaded then delete the locally saved dataset so that the new dataset can be made available locally
    if is_data_uploaded():
        os.remove("uploaded_data.csv")

    # deleting the old best model pickle file to start from scratch again for the new dataset
    if is_model_trained():
        os.remove("best_model.pkl")
    
    # for uploading CSV file
    file = st.file_uploader("Upload Data (CSV only)")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("uploaded_data.csv", index=None)
        # displaying the data on app after uploading file successfully
        st.dataframe(df)

# for Auto-EDA option
if choice == "Auto-EDA":
    # if data is uploaded only then proceed with profiling
    if is_data_uploaded():
        st.title("Automated Data Analysis")
        # generate profile report
        report = df.profile_report()
        # display profile report
        st_profile_report(report)
        # to download profile report as html file
        st.download_button("Download Report", report.to_html(), file_name="report.html", mime="text/html")

    # if data is not uploaded display a message asking the user to upload data
    else:
        st.title("Automated Data Analysis")
        st.info("Make sure you've uploaded the dataset!")
    
# for training option
if choice == "Train Models":
    # again if data is uploaded only then proceed towards training
    if is_data_uploaded():
        st.title("Automated ML Model Trainer")
        # to select the type of problem between regression and classification
        type = st.selectbox("Select the type of Problem", ["Regression", "Classification"])
        # to select the target column
        target = st.selectbox("Select the Target Variable", df.columns)

        # button to start training
        if st.button("Start Training"):
            # pycaret functions for regression models
            if type == "Regression":
                rsetup(df, target=target, verbose=False)
                setup_df = rpull()
                best_model = rcompare_models()
                rsave_model(best_model, "best_model")
                compare_df = rpull()

            # pycaret functions for classification models
            if type == "Classification":
                csetup(df, target=target, verbose=False)
                setup_df = cpull()
                best_model = ccompare_models()
                csave_model(best_model, "best_model")
                compare_df = cpull()

            # displaying all details regarding the ML models
            st.info("ML Model Experiment Settings")
            st.dataframe(setup_df)
            st.info("ML Models Comparison")
            st.dataframe(compare_df)
            st.info("Best Model Parameters")
            best_model
    
    # if data is not uploaded ask the user to upload it
    else:
        st.title("Automated ML Model Trainer")
        st.info("Make sure you've uploaded the dataset!")

# for download option
if choice == "Download Best Model":
    # if data is uploaded proceed further
    if is_data_uploaded():
        # if models were trained proceed further
        if is_model_trained():
            st.title("Download Best Model")
            # opening the file that was saved earlier so that user can download it and use it on his/her local machine
            with open("best_model.pkl", "rb") as file:
                st.download_button("Download Model", file, "trained_model.pkl")
        # if models are not trained yet ask the user to train models
        else:
            st.title("Download Best Model")
            st.info("Make sure you've trained the models!")
    # if data is not uploaded ask user to upload it
    else:
        st.title("Download Best Model")
        st.info("Make sure you've uploaded the dataset!")
