import streamlit as st
import snowflake.snowpark.functions as F
from snowflake.snowpark.types import DoubleType
import pandas as pd


session = st.session_state.session

st.subheader('Step 1: Load & Clean Data')

st.write("The diamonds dataset has been widely used in data science and machine learning. We will use it to demonstrate Snowflake's native data science transformers in terms of database functionality and Spark & Pandas comportablity, using non-synthetic and statistically appropriate data that is well known to the ML community.")

st.info("You can paste the below code into any Jupyter/Snowflake notebook.", icon="💡")

with st.expander("Libraries you need to import for this step"):

    st.code(f"""

        import snowflake.snowpark.functions as F
        from snowflake.snowpark.types import DoubleType

    """)

# setup our tabs 
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preview", "Load", "Show descriptive stats", "Data Cleaning", "Write back to warehouse"])

with tab1: 

    code_preview_data = '''
            # Grab file info and display it before loading

            session.sql("USE ML_HOL_DB;")
            session.sql("LS @DIAMONDS_ASSETS;"))
    '''

    st.code(code_preview_data)

    if st.button("Run the example", key=1):

        # Show the file before loading
        session.sql("USE ML_HOL_DB;")
        st.write(session.sql("LS @DIAMONDS_ASSETS;"))

with tab2:

    code_load_data = '''
        # Create a Snowpark DataFrame that is configured 
        # to load data from the CSV file
        # We can now infer schema from CSV files.

        diamonds_df = session.read.options(
            {
                "field_delimiter": ",",
                "field_optionally_enclosed_by": '"',
                "infer_schema": True,
                "parse_header": True
            }
        ).csv("@DIAMONDS_ASSETS")

        diamonds_df.head(10)
        '''

    st.code(code_load_data)

    if st.button("Run the example", key=2):

        with st.spinner('Wait for it...'):
            

            # Create a Snowpark DataFrame that is configured to load data from the CSV file
            # We can now infer schema from CSV files.
            diamonds_df = session.read.options({"field_delimiter": ",",
                                            "field_optionally_enclosed_by": '"',
                                                "infer_schema": True,
                                                "parse_header": True}).csv("@DIAMONDS_ASSETS")

            #save to Streamlit session to load across tabs 

            st.session_state['diamonds_df'] = diamonds_df

            st.write("Here's the first 10 rows of `diamonds_df:`")

            #convert to pandas dataframe to use the .head() function

            st.write(diamonds_df.to_pandas().head(10))


with tab3:
    code_descriptive_stats = '''
     # Look at descriptive stats on the DataFrame
    
    diamonds_df.describe()
    '''

    st.code(code_descriptive_stats)

    if st.button("Run the example", key=3):

        if 'diamonds_df' in st.session_state:
            diamonds_df = st.session_state.diamonds_df

            st.write(diamonds_df.describe())

        else: 
            st.write('Run the "Load" tab first to load your data!')

with tab4:

    code_data_cleaning = '''

    # make a new dataframe for our clean data
    # this step is optional but a good practice that makes it
    # easier to debug your code later on 
    diamonds_df_clean = diamonds_df
    
    # Let's make our column names uppercase and strip extra "s
    # Except TABLE is a reserved keyword in Snowflake 
    # We want to uppercase it but keep escaping this one
    colnames = {}

    for column in diamonds_df.columns:
        if column != '"table"':
            col_new = column.replace('"','').upper()
            colnames.update({column: col_new})
        else: 
            col_new = column.upper()
            colnames.update({column: col_new})


    diamonds_df_upper = diamonds_df.rename(colnames)

    # define a function that takes column values, 
    # converts them to upper case and 
    # replaces spaces with underscores 

    def fix_values(columnn):
            strip_spaces = F.regexp_replace(
                                F.col(columnn), 
                                '[^a-zA-Z0-9]+',
                                '_')
            return F.upper(strip_spaces)

    for col in ['CUT']: 
        diamonds_df_clean = diamonds_df_clean.with_column(col, fix_values(col))

    for colname in ['CARAT', 'X', 'Y', 'Z', 'DEPTH', 'TABLE_PCT']:
        diamonds_df_clean = diamonds_df_clean.with_column(colname, diamonds_df_clean[colname].cast(DoubleType()))


    # show the final product! 
    diamonds_df_clean.head(10)
    '''

    st.code(code_data_cleaning)
    
    if st.button("Run the example", key=4):

        if 'diamonds_df' in st.session_state:
           
            diamonds_df = st.session_state.diamonds_df

            colnames = {}

            for column in diamonds_df.columns:
                if column != '"table"':
                    col_new = column.replace('"','').upper()
                    colnames.update({column: col_new})
                else: 
                    col_new = 'TABLE_PCT'
                    colnames.update({column: col_new})
        

            diamonds_df_upper = diamonds_df.rename(colnames)

            def fix_values(columnn):
                strip_spaces = F.regexp_replace(
                                    F.col(columnn), 
                                    '[^a-zA-Z0-9]+',
                                    '_')
                return F.upper(strip_spaces)
            
            for col in ['CUT']: 
                diamonds_df_clean = diamonds_df_upper.with_column(col, fix_values(col))

            for colname in ['CARAT', 'X', 'Y', 'Z', 'DEPTH', 'TABLE_PCT']:
                diamonds_df_clean = diamonds_df_clean.with_column(colname, diamonds_df_clean[colname].cast(DoubleType()))

            st.session_state['diamonds_df_clean'] = diamonds_df_clean

            st.write("Here's the first 10 rows of our `diamonds_df_clean` dataframe:")

            st.write(diamonds_df_clean.to_pandas().head(10))

            # overwrites data with formatted version in actual db -- uncomment to fix any weirdness
            #diamonds_df_clean.write.mode('overwrite').save_as_table('diamonds')

        else: 
            st.write('Run the "Load" tab first to load your data!')

with tab5:

    code_write_table = '''
            # Persist diamonds_df in a table in our schema

            diamonds_df_clean.write.mode('overwrite').save_as_table('diamonds')
    '''

    st.code(code_write_table)

    if st.button("Run the example", key=5):

        input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.diamonds"

        test_df = session.table(input_tbl)

        if test_df:
            st.success('Saved table!')
        else: 
            st.write('Oops something went wrong! Please contact XX@yy.com to report an issue with this app')