import streamlit as st
import snowflake.snowpark.functions as F


tab1, tab2, tab3, tab4 = st.tabs(["Preview Data", "Load", "Show descriptive stats", "Clean up the data"])

with tab1: 

    code_preview_data = '''
            session.sql("USE ML_HOL_DB;")
            session.sql("LS @DIAMONDS_ASSETS;"))
    '''

    st.code(code_preview_data)

    if st.button("Run the example", key=1):

        # Show the file before loading
        st.session_state.session.sql("USE ML_HOL_DB;")
        st.write(st.session_state.session.sql("LS @DIAMONDS_ASSETS;"))

with tab2:

    code_load_data = '''
        # Create a Snowpark DataFrame that is configured to load data from the CSV file
        # We can now infer schema from CSV files.
        diamonds_df = session.read.options({"field_delimiter": ",",
                                        "field_optionally_enclosed_by": '"',
                                            "infer_schema": True,
                                            "parse_header": True}).csv("@DIAMONDS_ASSETS")

        diamonds_df.head(10)
        '''

    st.code(code_load_data)

    if st.button("Run the example", key=2):

        # Create a Snowpark DataFrame that is configured to load data from the CSV file
        # We can now infer schema from CSV files.
        diamonds_df = st.session_state.session.read.options({"field_delimiter": ",",
                                        "field_optionally_enclosed_by": '"',
                                            "infer_schema": True,
                                            "parse_header": True}).csv("@DIAMONDS_ASSETS")

        st.session_state['diamonds_df'] = diamonds_df

        st.write(diamonds_df.to_pandas().head(10))

with tab3:
    code_descriptive_stats = '''
    diamonds_df.describe()
    '''

    st.code(code_descriptive_stats)

    if st.button("Run the example", key=3):

        # Look at descriptive stats on the DataFrame
        st.write(st.session_state.diamonds_df.describe())

with tab4:

    code_data_cleaning = '''
    def fix_values(columnn):
            return F.upper(F.regexp_replace(F.col(columnn), '[^a-zA-Z0-9]+', '_'))

    for col in ['\"cut\"']: 
        st.session_state.diamonds_df = st.session_state.diamonds_df.with_column(col, fix_values(col))

    diamonds_df.head(10)
    '''

    st.code(code_data_cleaning)
    
    if st.button("Run the example", key=4):

        #st.write(st.session_state.diamonds_df.to_pandas().rename(str.replace("\"",""), axis='columns'))


        def fix_values(columnn):
            return F.upper(F.regexp_replace(F.col(columnn), '[^a-zA-Z0-9]+', '_'))

        for col in ['\"cut\"']: 
            st.session_state.diamonds_df = st.session_state.diamonds_df.with_column(col, fix_values(col))

        st.write(st.session_state.diamonds_df.to_pandas().head(10))
