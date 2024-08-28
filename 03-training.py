import streamlit as st

import json
import joblib


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from snowflake.ml.modeling.metrics.correlation import correlation

session = st.session_state.session

st.subheader("Step 3: Exploring data and training our model")

st.write(
    "We will illustrate how to train an XGBoost model with the diamonds dataset using the Snowpark ML Modeling API.", 
    "We also show how to do inference and manage models via Model Registry or as a UDF (See Appendix).", 
    "\n\n 😎 **Did you know?** The Snowpark ML Model API supports scikit-learn, xgboost, and lightgbm models."
)

st.info("You can paste the below code into any Jupyter/Snowflake notebook.", icon="💡")

with st.expander("Libraries you need to import for this step"):

    st.code(f"""

        import json
        import joblib


        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        from snowflake.ml.modeling.metrics.correlation import correlation

    """)

tab1,tab2, tab3, tab4 = st.tabs(["Load data + pre-processing pipeline", "Explore the data", "Built an XGBoost regression model","Find optimal model parameters"])


with tab1:

    st.write("👉 Note that we've reused our code from Step 2 here to populate `diamonds_df_clean`. If you are working in the same notebook from Step 2, you can skip this :) ")

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

    st.write("Now let's load the pipeline we created at the end of Step 2, so we don't have to rewrite our transformation logic.")

    code_load_data_pipeline = '''       
        # Load the preprocessing pipeline object from stage
        # to do this, we download the preprocessing_pipeline.joblib.gz 
        # file to the warehouse where our code is running, and then load it using joblib.
        session.file.get('@ML_HOL_ASSETS/preprocessing_pipeline.joblib.gz', '/tmp')
        PIPELINE_FILE = '/tmp/preprocessing_pipeline.joblib.gz'
        preprocessing_pipeline = joblib.load(PIPELINE_FILE)

        # Transform the data 
        transformed_diamonds_df = preprocessing_pipeline.fit(diamonds_df).transform(diamonds_df)
    '''

    st.code(code_load_data_pipeline)

    if st.button("Run the example", key=1):

        with st.status("Wait for it...", expanded=True) as status:

            DEMO_TABLE = 'diamonds' 
            input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{DEMO_TABLE}"

            # Load in the data
            diamonds_df = session.table(input_tbl)
            st.session_state.diamonds_df = diamonds_df

            st.success('Data loaded!', icon='✅')

            # Categorize all the features for modeling
            #CATEGORICAL_COLUMNS = ["CUT", "COLOR", "CLARITY"]
            #CATEGORICAL_COLUMNS_OE = ["CUT_OE", "COLOR_OE", "CLARITY_OE"] # To name the ordinal encoded columns
            #NUMERICAL_COLUMNS = ["CARAT", "DEPTH", "TABLE_PCT", "X", "Y", "Z"]

            #LABEL_COLUMNS = ['PRICE']
            #OUTPUT_COLUMNS = ['PREDICTED_PRICE']

            # Load the preprocessing pipeline object from stage
            # to do this, we download the preprocessing_pipeline.joblib.gz 
            # file to the warehouse where our code is running, and then load it using joblib.
            session.file.get('@ML_HOL_ASSETS/preprocessing_pipeline.joblib.gz', '/tmp')
            PIPELINE_FILE = '/tmp/preprocessing_pipeline.joblib.gz'
            preprocessing_pipeline = joblib.load(PIPELINE_FILE)

            transformed_diamonds_df = preprocessing_pipeline.fit(diamonds_df).transform(diamonds_df)
            st.session_state.transformed_diamonds_df = transformed_diamonds_df
            
            st.success('Data transformed!', icon='✅')

            status.update(
                label="Pre-processing complete!", state="complete", expanded=False
            )
with tab2: 


    if 'transformed_diamonds_df' in st.session_state:
        transformed_diamonds_df = st.session_state.transformed_diamonds_df

    st.write("Now that we've transformed our features, let's calculate the correlation using Snowpark ML's `correlation()` function between each pair to better understand their relationships.")

    code_correlations = '''

        corr_diamonds_df = correlation(df=transformed_diamonds_df)
        corr_diamonds_df # This is a Pandas DataFrame

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_diamonds_df, dtype=bool))

        # Create a heatmap with the features
        plot = plt.figure(figsize=(7, 7))
        heatmap = sns.heatmap(corr_diamonds_df, mask=mask, cmap="YlGnBu", annot=True, vmin=-1, vmax=1)
        st.pyplot(plot)
            
    '''

    st.code(code_correlations)

    if st.button("Run some correlations", key=2):

        try:
            corr_diamonds_df = correlation(df=transformed_diamonds_df)
            #corr_diamonds_df # This is a Pandas DataFrame

            # Generate a mask for the upper triangle
            mask = np.triu(np.ones_like(corr_diamonds_df, dtype=bool))

            # Create a heatmap with the features
            plot = plt.figure(figsize=(7, 7))
            heatmap = sns.heatmap(corr_diamonds_df, mask=mask, cmap="YlGnBu", annot=True, vmin=-1, vmax=1)
            st.pyplot(plot)

        except:
            st.warning('Oops! You need to load some data first!', icon="🤭")



    st.write(
        "We note that CARAT and PRICE are highly correlated, which makes sense! Let's take a look at their relationship a bit closer.",
        "\n\n👉 You will have to convert your Snowpark DF to a Pandas DF in order to use matplotlib & seaborn."
    )

    code_scatter='''

        # Set up a plot to look at CARAT and PRICE
        counts = transformed_diamonds_df.to_pandas().groupby(['PRICE', 'CARAT']).size().reset_index(name='Count')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.scatterplot(data=counts, x='CARAT', y='PRICE', size='Count', markers='o', alpha=0.75)
        ax.grid(axis='y')

        # The relationship is not linear - it appears exponential which makes sense given the rarity of the large diamonds
        sns.move_legend(ax, "upper left")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
    '''

    st.code(code_scatter)
    
    if st.button("Show a scatterplot of the data", key=3):


        # Set up a plot to look at CARAT and PRICE
        counts = transformed_diamonds_df.to_pandas().groupby(['PRICE', 'CARAT']).size().reset_index(name='Count')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax = sns.scatterplot(data=counts, x='CARAT', y='PRICE', size='Count', markers='o', alpha=0.75)
        ax.grid(axis='y')

        # The relationship is not linear - it appears exponential which makes sense given the rarity of the large diamonds
        sns.move_legend(ax, "upper left")
        sns.despine(left=True, bottom=True)
        st.pyplot(fig)
