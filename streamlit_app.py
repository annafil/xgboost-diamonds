import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.types import StructType, StructField, DoubleType, StringType
import os
from streamlit.components.v1 import html

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

session = Session.builder.configs(connection_parameters).create()

st.session_state['session'] = session

session.sql_simplifier_enabled = True

snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
snowpark_version = VERSION

with st.sidebar: 
    with st.expander("Inspect the app connection"):

        st.caption("Connection Established with the following parameters:")

        st.code(f"""

        PySnowpark: {snowpark_version[0], snowpark_version[1], snowpark_version[2] }
        Current Schema: {session.get_current_schema()}
        Snowflake version: {snowflake_environment[0][1]}
        """)

        # uncomment and add in above code block to help debug your connection 
        # User: {snowflake_environment[0][0]}
        # Role: {session.get_current_role()}
        # Database: {session.get_current_database()}
        # Warehouse: {session.get_current_warehouse()}



pages = {
    "🎈+ ❄️ : XGBoost prediction in Python": [
        st.Page("overview.py", title="Overview"),
        st.Page("00-Snowflake-setup.py", title="Step 0: Set up"),
        st.Page("01-data-loading-cleaning.py", title="Step 1: Load & Clean Data"),
        st.Page("02-feature-transformation.py", title="Step 2: Feature Transformation"),
        st.Page("03-training.py", title="Step 3: Training!")
    ]
}

pg = st.navigation(pages)
pg.run()


# Define your javascript
ga_js = """

    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-MDTNPSFEW7"></script>
    <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', 'G-MDTNPSFEW7');
    </script>

"""

# Wrapt the javascript as html code
ga_html = f"<script>{ga_js}</script>"

# load cookie
html(ga_html)