import streamlit as st

import snowflake.ml.modeling.preprocessing as snowml
from snowflake.snowpark.types import DecimalType
from snowflake.ml.modeling.pipeline import Pipeline

import json
import joblib

import numpy as np



st.subheader("Step 2: Feature Transformations")

st.write(
    "In this section, we will walk through a few transformations that are included in the Snowpark ML Preprocessing API.", 
    "We will also build a preprocessing pipeline to be used in the ML modeling notebook.", 
    "\n\n 😎 **Fun fact:** All feature transformations using Snowpark ML are distributed operations -- this means you are using the power of a cloud environment in the backend, no matter where you write your code."
)

st.info("You can paste the below code into any Jupyter/Snowflake notebook.", icon="💡")

with st.expander("Libraries you need to import for this step"):

    st.code(f"""

        import snowflake.ml.modeling.preprocessing as snowml
        from snowflake.snowpark.types import DecimalType
        from snowflake.ml.modeling.pipeline import Pipeline

        import json
        import joblib

        import numpy as np

    """)

session = st.session_state.session

tab1,tab2, tab3 = st.tabs(["Load cleaned data","Transform features", "Build a pipeline"])

with tab1: 

    code_load_clean_data = '''

        # Specify the table name where we stored the diamonds dataset
        # Change this only if you named your table something else 
        # in the data ingest step
        DEMO_TABLE = 'diamonds'
        input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{DEMO_TABLE}"

         # Load table data into a DataFrame
        diamonds_df_clean = session.table(input_tbl)

        diamonds_df_clean.head(10)
    '''

    st.code(code_load_clean_data)

    if st.button("Run the example", key=1):

        with st.spinner('Wait for it...'):
            # Specify the table name where we stored the diamonds dataset
            # **nChange this only if you named your table something else in the data ingest notebook **
            DEMO_TABLE = 'diamonds'
            input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{DEMO_TABLE}"

            # First, we read in the data from a Snowflake table into a Snowpark DataFrame
            diamonds_df_clean = session.table(input_tbl)

            st.session_state['diamonds_df_clean'] = diamonds_df_clean

            st.write(diamonds_df_clean.to_pandas().head(10))

with tab2: 
    st.write(
            "We will illustrate a few of the transformation functions here, but the rest can be found in the documentation.",
            "\n\n Let's use the MinMaxScaler to normalize the CARAT column."
    )

    code_minmaxscaler = '''
        # Normalize the CARAT column
        snowml_mms = snowml.MinMaxScaler(input_cols=["CARAT"], output_cols=["CARAT_NORM"])
        normalized_diamonds_df = snowml_mms.fit(diamonds_df).transform(diamonds_df)

        # Reduce the number of decimals
        new_col = normalized_diamonds_df.col("CARAT_NORM").cast(DecimalType(7, 6))
        normalized_diamonds_df = normalized_diamonds_df.with_column("CARAT_NORM", new_col)

    '''

    st.code(code_minmaxscaler)

    if st.button("Run the example", key=2):

        if 'diamonds_df_clean' in st.session_state:
            diamonds_df_clean = st.session_state.diamonds_df_clean

            with st.spinner('Wait for it...'):
                #try:
                # Normalize the CARAT column
                snowml_mms = snowml.MinMaxScaler(input_cols=["CARAT"], output_cols=["CARAT_NORM"])
                normalized_diamonds_df = snowml_mms.fit(diamonds_df_clean).transform(diamonds_df_clean)

                # Reduce the number of decimals
                new_col = normalized_diamonds_df.col("CARAT_NORM").cast(DecimalType(7, 6))
                normalized_diamonds_df = normalized_diamonds_df.with_column("CARAT_NORM", new_col)

                # Save to reuse elsewhere
                st.session_state.normalized_diamonds_df = normalized_diamonds_df

                st.write(normalized_diamonds_df)

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

    if st.button("Run the example", key=3):

        if 'normalized_diamonds_df' in st.session_state:
            normalized_diamonds_df = st.session_state.normalized_diamonds_df

            with st.spinner('Wait for it...'):

                try:
                    categories = {
                        "CUT": np.array(["IDEAL", "PREMIUM", "VERY_GOOD", "GOOD", "FAIR"]),
                        "CLARITY": np.array(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]),
                    }
                    snowml_oe = snowml.OrdinalEncoder(input_cols=["CUT", "CLARITY"], output_cols=["CUT_OE", "CLARITY_OE"], categories=categories)
                    ord_encoded_diamonds_df = snowml_oe.fit(normalized_diamonds_df).transform(normalized_diamonds_df)

                    # Show the encoding
                    st.write(snowml_oe._state_pandas)

                    st.write(ord_encoded_diamonds_df)

                    st.session_state.ord_encoded_diamonds_df = ord_encoded_diamonds_df

                except: 
                    st.write('Run MinMaxScaler above first!')

with tab3: 

    st.write(
        "Finally, we can also build out a full preprocessing Pipeline.",
        "This will be useful for both the training we do in Step 3 as well as to productionize our model with standarized feature transformations."
        )

    code_build_pipeline = '''

        # Categorize all the features for processing
        CATEGORICAL_COLUMNS = ["CUT", "COLOR", "CLARITY"]
        CATEGORICAL_COLUMNS_OE = ["CUT_OE", "COLOR_OE", "CLARITY_OE"] # To name the ordinal encoded columns
        NUMERICAL_COLUMNS = ["CARAT", "DEPTH", "TABLE_PCT", "X", "Y", "Z"]

        categories = {
            "CUT": np.array(["IDEAL", "PREMIUM", "VERY_GOOD", "GOOD", "FAIR"]),
            "CLARITY": np.array(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]),
            "COLOR": np.array(['D', 'E', 'F', 'G', 'H', 'I', 'J']),
        }

        # Build the pipeline
        preprocessing_pipeline = Pipeline(
            steps=[
                    (
                        "OE",
                        snowml.OrdinalEncoder(
                            input_cols=CATEGORICAL_COLUMNS,
                            output_cols=CATEGORICAL_COLUMNS_OE,
                            categories=categories,
                        )
                    ),
                    (
                        "MMS",
                        snowml.MinMaxScaler(
                            clip=True,
                            input_cols=NUMERICAL_COLUMNS,
                            output_cols=NUMERICAL_COLUMNS,
                        )
                    )
            ]
        )

        PIPELINE_FILE = '/tmp/preprocessing_pipeline.joblib'
        joblib.dump(preprocessing_pipeline, PIPELINE_FILE) # We are just pickling it locally first

        transformed_diamonds_df = preprocessing_pipeline.fit(diamonds_df_clean).transform(diamonds_df_clean)
        transformed_diamonds_df.show()

        # You can also save the pickled object into the stage we created earlier for deployment
        session.file.put(PIPELINE_FILE, "@ML_HOL_ASSETS", overwrite=True)

    '''

    st.code(code_build_pipeline)

    if st.button("Run the example", key=4):

        if st.session_state.diamonds_df_clean: 
            diamonds_df_clean = st.session_state.diamonds_df_clean
            # Categorize all the features for processing
            CATEGORICAL_COLUMNS = ["CUT", "COLOR", "CLARITY"]
            CATEGORICAL_COLUMNS_OE = ["CUT_OE", "COLOR_OE", "CLARITY_OE"] # To name the ordinal encoded columns
            NUMERICAL_COLUMNS = ["CARAT", "DEPTH", "TABLE_PCT", "X", "Y", "Z"]

            categories = {
                "CUT": np.array(["IDEAL", "PREMIUM", "VERY_GOOD", "GOOD", "FAIR"]),
                "CLARITY": np.array(["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"]),
                "COLOR": np.array(['D', 'E', 'F', 'G', 'H', 'I', 'J']),
            }

            # Build the pipeline
            preprocessing_pipeline = Pipeline(
                steps=[
                        (
                            "OE",
                            snowml.OrdinalEncoder(
                                input_cols=CATEGORICAL_COLUMNS,
                                output_cols=CATEGORICAL_COLUMNS_OE,
                                categories=categories,
                            )
                        ),
                        (
                            "MMS",
                            snowml.MinMaxScaler(
                                clip=True,
                                input_cols=NUMERICAL_COLUMNS,
                                output_cols=NUMERICAL_COLUMNS,
                            )
                        )
                ]
            )

            PIPELINE_FILE = '/tmp/preprocessing_pipeline.joblib'
            joblib.dump(preprocessing_pipeline, PIPELINE_FILE) # We are just pickling it locally first

            transformed_diamonds_df = preprocessing_pipeline.fit(diamonds_df_clean).transform(diamonds_df_clean)
            st.write(
                "Here's our transformed dataframe using Snowflake's pipelines!", 
                transformed_diamonds_df.to_pandas().head(10))
            