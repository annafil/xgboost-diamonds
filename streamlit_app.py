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

st.title("🎈 Snowpark Python ML Demo")
st.write(
    "Through this quickstart guide, you will explore what's new in Snowflake for Machine Learning. You will build an end to end ML workflow from feature engineering to model training and deployment using Snowflake ML in Streamlit."
)

session = Session.builder.configs(connection_parameters).create()

st.session_state['session'] = session

session.sql_simplifier_enabled = True

snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
snowpark_version = VERSION

with st.sidebar: 
    with st.expander("Connection details"):
        st.write('Connection Established with the following parameters:')
        st.write('User                        : ' , snowflake_environment[0][0])
        st.write('Snowpark for Python version : ', snowpark_version[0], snowpark_version[1], snowpark_version[2]) 
        st.write('Role                        : ', session.get_current_role())
        st.write('Database                    : ', session.get_current_database())
        st.write('Schema                      : ', session.get_current_schema())
        st.write('Warehouse                    : ', session.get_current_warehouse())
        st.write('Snowflake version            : ', snowflake_environment[0][1])




pages = {
    "Tutorial": [
        st.Page("01-data-ingestion.py", title="Step 1: Data Ingestion"),
        st.Page("02-feature-transformation.py", title="Step 2: Feature Transformation"),
    ]
}

pg = st.navigation(pages)
pg.run()