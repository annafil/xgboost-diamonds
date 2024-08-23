import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.types import StructType, StructField, DoubleType, StringType
import snowflake.snowpark.functions as F
import os


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
session = Session.builder.configs(connection_parameters).create()

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

session.sql_simplifier_enabled = True

snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
snowpark_version = VERSION

with st.popover("Connection details"):
    st.write('Connection Established with the following parameters:')
    st.write('User                        : ' , snowflake_environment[0][0])
    st.write('Snowpark for Python version : ', snowpark_version[0], snowpark_version[1], snowpark_version[2]) 
    st.write('Role                        : ', session.get_current_role())
    st.write('Database                    : ', session.get_current_database())
    st.write('Schema                      : ', session.get_current_schema())
    st.write('Warehouse                    : ', session.get_current_warehouse())
    st.write('Snowflake version            : ', snowflake_environment[0][1])


# Show the file before loading
session.sql("USE ML_HOL_DB;")
session.sql("LS @DIAMONDS_ASSETS;")

# Create a Snowpark DataFrame that is configured to load data from the CSV file
# We can now infer schema from CSV files.
diamonds_df = session.read.options({"field_delimiter": ",",
                                 "field_optionally_enclosed_by": '"',
                                    "infer_schema": True,
                                    "parse_header": True}).csv("@DIAMONDS_ASSETS")

st.write(diamonds_df.to_pandas().head(10))

# Look at descriptive stats on the DataFrame
st.write(diamonds_df.describe())