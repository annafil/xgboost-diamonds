import streamlit as st
import snowflake.ml.modeling.preprocessing as snowml


st.subheader("Step 2: Feature Transformations")

st.write(
    "In this section, we will walk through a few transformations that are included in the Snowpark ML Preprocessing API.", 
    "We will also build a preprocessing pipeline to be used in the ML modeling notebook.", 
    "\n\n **Note:** All feature transformations using Snowpark ML are distributed operations in the same way that Snowpark DataFrame operations are."
)

session = st.session_state.session

# Specify the table name where we stored the diamonds dataset
# **nChange this only if you named your table something else in the data ingest notebook **
DEMO_TABLE = 'diamonds'
input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{DEMO_TABLE}"

st.write(input_tbl)

# First, we read in the data from a Snowflake table into a Snowpark DataFrame
diamonds_df = session.table(input_tbl)

st.dataframe(diamonds_df)

st.write(
        "Feature Transformations",
        "\n\n We will illustrate a few of the transformation functions here, but the rest can be found in the documentation.",
        "\n\n Let's use the MinMaxScaler to normalize the CARAT column."
)

# Normalize the CARAT column
snowml_mms = snowml.MinMaxScaler(input_cols=["CARAT"], output_cols=["CARAT_NORM"])
normalized_diamonds_df = snowml_mms.fit(diamonds_df).transform(diamonds_df)

# Reduce the number of decimals
new_col = normalized_diamonds_df.col("CARAT_NORM").cast(DecimalType(7, 6))
normalized_diamonds_df = normalized_diamonds_df.with_column("CARAT_NORM", new_col)

st.write(normalized_diamonds_df)

