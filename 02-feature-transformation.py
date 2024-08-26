import streamlit as st
import snowflake.ml.modeling.preprocessing as snowml
from snowflake.snowpark.types import DecimalType
import numpy as np


st.subheader("Step 2: Feature Transformations")

st.write(
    "In this section, we will walk through a few transformations that are included in the Snowpark ML Preprocessing API.", 
    "We will also build a preprocessing pipeline to be used in the ML modeling notebook.", 
    "\n\n **Note:** All feature transformations using Snowpark ML are distributed operations in the same way that Snowpark DataFrame operations are."
)

with st.expander("Libraries you need to import for this step"):

    st.code(f"""

        import snowflake.ml.modeling.preprocessing as snowml
        from snowflake.snowpark.types import DecimalType
        import numpy as np

    """)

session = st.session_state.session

tab1,tab2 = st.tabs(["Load cleaned data","Transform features"])

with tab1: 

    code_load_clean_data = '''

        # Specify the table name where we stored the diamonds dataset
        # Change this only if you named your table something else 
        # in the data ingest step
        DEMO_TABLE = 'diamonds'
        input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{DEMO_TABLE}"

         # Load table data into a DataFrame
        diamonds_df = session.table(input_tbl)

        diamonds_df.head(10)
    '''

    st.code(code_load_clean_data)

    if st.button("Run the example", key=1):


        # Specify the table name where we stored the diamonds dataset
        # **nChange this only if you named your table something else in the data ingest notebook **
        DEMO_TABLE = 'diamonds'
        input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{DEMO_TABLE}"

        # First, we read in the data from a Snowflake table into a Snowpark DataFrame
        diamonds_df = session.table(input_tbl)

        st.write(diamonds_df.to_pandas().head(10))


with tab2: 
    st.write(
            "We will illustrate a few of the transformation functions here, but the rest can be found in the documentation.",
            "\n\n Let's use the MinMaxScaler to normalize the CARAT column."
    )

    code_minmaxscaler = '''
        # Normalize the CARAT column
        snowml_mms = snowml.MinMaxScaler(input_cols=["\"carat\""], output_cols=["carat_norm"])
        normalized_diamonds_df = snowml_mms.fit(diamonds_df).transform(diamonds_df)

        # Reduce the number of decimals
        new_col = normalized_diamonds_df.col("carat_norm").cast(DecimalType(7, 6))
        normalized_diamonds_df = normalized_diamonds_df.with_column("carat_norm", new_col)

    '''

    st.code(code_minmaxscaler)

    st.write("We can also use the OrdinalEncoder to transform COLOR and CLARITY from categorical to numerical values so they are more meaningful.")

    code_ordinalencoder ='''

        categories = {
            "CUT": np.array(["IDEAL", "PREMIUM", "VERY_GOOD", "GOOD", "FAIR"]),
            "CLARITY": np.array(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]),
        }
        snowml_oe = snowml.OrdinalEncoder(input_cols=["CUT", "CLARITY"], output_cols=["CUT_OE", "CLARITY_OE"], categories=categories)
        ord_encoded_diamonds_df = snowml_oe.fit(normalized_diamonds_df).transform(normalized_diamonds_df)

        # Show the encoding
        print(snowml_oe._state_pandas)

        ord_encoded_diamonds_df.show()

    '''

    st.code(code_ordinalencoder)


    if st.button("Run the example", key=2):

        if 'diamonds_df' in st.session_state:
            diamonds_df = st.session_state.diamonds_df

        try:
            # Normalize the CARAT column
            snowml_mms = snowml.MinMaxScaler(input_cols=["\"carat\""], output_cols=["carat_norm"])
            normalized_diamonds_df = snowml_mms.fit(diamonds_df).transform(diamonds_df)

            # Reduce the number of decimals
            new_col = normalized_diamonds_df.col("carat_norm").cast(DecimalType(7, 6))
            normalized_diamonds_df = normalized_diamonds_df.with_column("carat_norm", new_col)

            st.write(normalized_diamonds_df)

            categories = {
                "\"cut\"": np.array(["Ideal", "Premium", "Very Good", "Good", "Fair"]),
                "\"clarity\"": np.array(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]),
            }
            snowml_oe = snowml.OrdinalEncoder(input_cols=["\"cut\"", "\"clarity\""], output_cols=["CUT_OE", "CLARITY_OE"], categories=categories)
            #ord_encoded_diamonds_df = snowml_oe.fit(normalized_diamonds_df).transform(normalized_diamonds_df)

            # Show the encoding
            #st.write(snowml_oe._state_pandas)

            #st.write(ord_encoded_diamonds_df)


        except: 
            st.write('Run the load example first!')



