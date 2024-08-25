import streamlit as st
import snowflake.snowpark.functions as F
from snowflake.snowpark.types import DoubleType


session = st.session_state.session

st.subheader('Step 1: Data Ingestion')

st.write("The diamonds dataset has been widely used in data science and machine learning. We will use it to demonstrate Snowflake's native data science transformers in terms of database functionality and Spark & Pandas comportablity, using non-synthetic and statistically appropriate data that is well known to the ML community.")

st.info("You can paste the below code into any Jupyter/Snowflake notebook.", icon="💡")

# setup our tabs 
tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Load", "Show descriptive stats", "Data Cleaning"])

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
    # define a function that takes column values, 
    # converts them to upper case and 
    # replaces spaces with underscores 

    def fix_values(columnn):
            strip_spaces = F.regexp_replace(
                                F.col(columnn), 
                                '[^a-zA-Z0-9]+',
                                '_')
            return F.upper(strip_spaces)

    # our file is loaded with extra \" characters
    # we need to escape them to load our column 

    for col in ['\\"cut\\"']: 
        diamonds_df = diamonds_df.with_column(col, fix_values(col))

    # show the final product! 

    diamonds_df.head(10)
    '''

    st.code(code_data_cleaning)
    
    if st.button("Run the example", key=4):

        if 'diamonds_df' in st.session_state:
           
            diamonds_df = st.session_state.diamonds_df

            with st.spinner('Wait for it...'):

                #st.write(diamonds_df.to_pandas().rename(str.replace("\"",""), axis='columns'))

                def fix_values(columnn):
                    strip_spaces = F.regexp_replace(
                                        F.col(columnn), 
                                        '[^a-zA-Z0-9]+',
                                        '_')
                    return F.upper(strip_spaces)
                
                for col in ['\"cut\"']: 
                    diamonds_df = diamonds_df.with_column(col, fix_values(col))

                #for colname in diamonds_df.columns: 
                #    st.write(colname)

                for colname in ["\"carat\"", "\"x\"", "\"y\"", "\"z\"", "\"depth\"", "\"table\""]:
                    diamonds_df = diamonds_df.with_column(colname, diamonds_df[colname].cast(DoubleType()))

                st.write(diamonds_df.to_pandas().head(10))

                diamonds_df.write.mode('overwrite').save_as_table('diamonds')

                st.success('Saved table!')
        else: 
            st.write('Run the "Load" tab first to load your data!')
