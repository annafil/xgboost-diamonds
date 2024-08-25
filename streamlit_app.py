import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.types import StructType, StructField, DoubleType, StringType
import os


# config 

connection_parameters = {
   "account":st.secrets['DB_ACCOUNT'],
   "user": st.secrets['DB_USER'],
   "password": st.secrets['DB_PASSWORD'],
   "role": st.secrets['DB_ROLE'],  # optional
   "warehouse": st.secrets['DB_WAREHOUSE'],  # optional
   "database": st.secrets['DB_DATABASE'],  # optional 
   "schema": "ML_HOL_SCHEMA",
   "client_session_keep_alive": True

}

st.header("🎈+ ❄️ : XGBoost Prediction in Python")
st.write(
    "This interactive 🎈Streamlit example introduces you to Machine Learning in Python on ❄️Snowflake:",
     "\n\n - ✅ At the end of this demo, you will be able to build a simple, end to end ML workflow in Python from ingestion to a dynamic interface you can use to predict diamond prices. You will be using XGBoost and the classic [Diamonds dataset](https://ggplot2.tidyverse.org/reference/diamonds.html).",
     "\n\n - 💡 You can run this example inside a Snowflake Notebook or a classic Jypyter Notebook on your machine. You will build an interactive application using 🎈 Streamlit at the end.",
    "\n\n - 👉 If you prefer to download the code to run on your own, just clone or fork this [GitHub Repo](https://github.com/Snowflake-Labs/sfguide-intro-to-machine-learning-with-snowflake-ml-for-python/)."
)

session = Session.builder.configs(connection_parameters).create()

st.session_state['session'] = session

session.sql_simplifier_enabled = True

snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
snowpark_version = VERSION

with st.sidebar: 
    with st.expander("Inspect the app connection"):
        
        st.write('Connection Established with the following parameters:')
        st.write('User                        : ' , snowflake_environment[0][0])
        st.write('Snowpark for Python version : ', snowpark_version[0], snowpark_version[1], snowpark_version[2]) 
        st.write('Role                        : ', session.get_current_role())
        st.write('Database                    : ', session.get_current_database())
        st.write('Schema                      : ', session.get_current_schema())
        st.write('Warehouse                    : ', session.get_current_warehouse())
        st.write('Snowflake version            : ', snowflake_environment[0][1])




pages = {
    "🎈+ ❄️ : XGboost prediction in Python": [
        st.Page("00-Snowflake-setup.py", title="Step 0: Set up"),
        st.Page("01-data-ingestion.py", title="Step 1: Data Ingestion"),
        st.Page("02-feature-transformation.py", title="Step 2: Feature Transformation"),
    ]
}

pg = st.navigation(pages)
pg.run()