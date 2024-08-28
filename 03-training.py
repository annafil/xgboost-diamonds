import streamlit as st

import json
import joblib


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from snowflake.ml.modeling.metrics.correlation import correlation


session = st.session_state.session


tab1,tab2, tab3, tab4 = st.tabs(["Load data + pre-processing pipeline", "Explore the data", "Built an XGBoost regression model","Find optimal model parameters"])


with tab1:

    code_load_data_pipeline = '''
        # Specify the table name where we stored the diamonds dataset
        # Change this only if you named your table something else in Step 1
        DEMO_TABLE = 'diamonds' 
        input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{DEMO_TABLE}"

        # Load in the data
        diamonds_df = session.table(input_tbl)
        diamonds_df.show()

        # Categorize all the features for modeling
        CATEGORICAL_COLUMNS = ["CUT", "COLOR", "CLARITY"]
        CATEGORICAL_COLUMNS_OE = ["CUT_OE", "COLOR_OE", "CLARITY_OE"] # To name the ordinal encoded columns
        NUMERICAL_COLUMNS = ["CARAT", "DEPTH", "TABLE_PCT", "X", "Y", "Z"]

        LABEL_COLUMNS = ['PRICE']
        OUTPUT_COLUMNS = ['PREDICTED_PRICE']

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

    if st.toggle("Run the example", key=1):

        with st.status("Wait for it...", expanded=True) as status:

            DEMO_TABLE = 'diamonds' 
            input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{DEMO_TABLE}"

            # Load in the data
            diamonds_df = session.table(input_tbl)
            st.session_state.diamonds_df = diamonds_df

            st.success('Data loaded!', icon='✅')

            # Categorize all the features for modeling
            CATEGORICAL_COLUMNS = ["CUT", "COLOR", "CLARITY"]
            CATEGORICAL_COLUMNS_OE = ["CUT_OE", "COLOR_OE", "CLARITY_OE"] # To name the ordinal encoded columns
            NUMERICAL_COLUMNS = ["CARAT", "DEPTH", "TABLE_PCT", "X", "Y", "Z"]

            LABEL_COLUMNS = ['PRICE']
            OUTPUT_COLUMNS = ['PREDICTED_PRICE']

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

    code_explore_data='''

        corr_diamonds_df = correlation(df=transformed_diamonds_df)
        corr_diamonds_df # This is a Pandas DataFrame

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_diamonds_df, dtype=bool))

        # Create a heatmap with the features
        plot = plt.figure(figsize=(7, 7))
        heatmap = sns.heatmap(corr_diamonds_df, mask=mask, cmap="YlGnBu", annot=True, vmin=-1, vmax=1)
        st.pyplot(plot)

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

    col1, col2 = st.columns(2)

    with col1: 

        if 'transformed_diamonds_df' in st.session_state:
            transformed_diamonds_df = st.session_state.transformed_diamonds_df


        if st.toggle("Run some correlations", key=2):

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

    with col2:

        if st.toggle("Show a scatterplot of the data", key=3):


            # Set up a plot to look at CARAT and PRICE
            counts = transformed_diamonds_df.to_pandas().groupby(['PRICE', 'CARAT']).size().reset_index(name='Count')

            fig, ax = plt.subplots(figsize=(10, 6))
            ax = sns.scatterplot(data=counts, x='CARAT', y='PRICE', size='Count', markers='o', alpha=0.75)
            ax.grid(axis='y')

            # The relationship is not linear - it appears exponential which makes sense given the rarity of the large diamonds
            sns.move_legend(ax, "upper left")
            sns.despine(left=True, bottom=True)
            st.pyplot(fig)
