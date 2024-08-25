import streamlit as st

st.subheader('Step 0: Setting up this example in Snowflake')

tab1, tab2 = st.tabs(["Setup your database and data sources", "Setup your Snowflake connection"])

with tab1:

    full_access = "I have access to create new warehouses and databases"
    limited_access = "I only have access to create and modify schemas"
    dont_know = "I'm not sure"

    access = st.radio(
        "What kind of warehouse access do you have?",
        [full_access, limited_access, dont_know],
        captions=[
            "e.g. if you signed up for a [Snowflake Free Trial](https://signup.snowflake.com/?utm_source=streamlit) or you have [rights to create new objects](https://docs.snowflake.com/en/user-guide/security-access-control-considerations#avoid-using-the-accountadmin-role-to-create-objects) on your Snowflake account",
            "most common if you are trying this example on your existing company's account",
            "select this option to learn how to check your access level",
        ],
        index=None,
    )

    
    if access:
        st.info("You can copy and paste this code into a [Snowflake Notebook SQL cell](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-develop-run#create-a-new-cell), [Snowflake SQL Workshet](https://docs.snowflake.com/en/user-guide/ui-snowsight-query) or the [Classic Console](https://docs.snowflake.com/en/user-guide/ui-using). You can also save the below code into a `setup.sql` file and run it using [SnowCLI](https://docs.snowflake.com/en/user-guide/snowsql-use#running-while-connecting-f-connection-parameter). If you choose to use SnowCLI, make sure you [setup your Snowflake connection](https://quickstarts.snowflake.com/guide/getting-started-with-snowflake-cli/index.html#2) first.", 
                icon="💡")


    if access==full_access:

        setup_code = '''

        -- Use a role here that allows you to create a warehouse
        USE ROLE SYSADMIN;

        -- Create a new warehouse for this example
        -- By default, this creates an XS standard warehouse
        CREATE OR REPLACE WAREHOUSE ML_HOL_WH; 

        -- Create a new database for this example 
        CREATE OR REPLACE DATABASE ML_HOL_DB;

        -- Create a new schema for this example 
        CREATE OR REPLACE SCHEMA ML_HOL_SCHEMA;

        -- Create a new stage for storing model assets 
        CREATE OR REPLACE STAGE ML_HOL_ASSETS;

        -- Create csv format on your schema
        CREATE FILE FORMAT IF NOT EXISTS ML_HOL_DB.ML_HOL_SCHEMA.CSVFORMAT 
            SKIP_HEADER = 1 
            TYPE = 'CSV';

        -- Create external stage with the csv format to stage the diamonds dataset
        CREATE STAGE IF NOT EXISTS ML_HOL_DB.ML_HOL_SCHEMA.DIAMONDS_ASSETS 
            FILE_FORMAT = ML_HOL_DB.ML_HOL_SCHEMA.CSVFORMAT 
            URL = 's3://sfquickstarts/intro-to-machine-learning-with-snowpark-ml-for-python/diamonds.csv';

        -- Test that everything worked! 
        
        LS @DIAMONDS_ASSETS;

        '''

        st.code(setup_code, language='sql')



    elif access==limited_access:

        setup_code_v2 = '''

        -- Use a role here that allows you to create and modify a schema
        USE ROLE YOUR_ROLE;

        -- Use an existing warehouse
        -- Replace YOUR_WAREHOUSE with the name of a warehouse you have access to 
        USE WAREHOUSE YOUR_WAREHOUSE; 

        -- Use an existing database. Replace value with your own! 
        USE DATABASE YOUR_DATABASE;

        -- Create a new schema for this example
        CREATE OR REPLACE SCHEMA ML_HOL_SCHEMA;

        -- Create a new stage for storing model assets 
        CREATE OR REPLACE STAGE ML_HOL_ASSETS;

        -- Create csv format on your schema
        CREATE FILE FORMAT IF NOT EXISTS YOUR_DATABASE.ML_HOL_SCHEMA.CSVFORMAT 
            SKIP_HEADER = 1 
            TYPE = 'CSV';

        -- Create external stage with the csv format to stage the diamonds dataset
        CREATE STAGE IF NOT EXISTS YOUR_DATABASE.ML_HOL_SCHEMA.DIAMONDS_ASSETS 
            FILE_FORMAT = YOUR_DATABASE.ML_HOL_SCHEMA.CSVFORMAT 
            URL = 's3://sfquickstarts/intro-to-machine-learning-with-snowpark-ml-for-python/diamonds.csv';

        -- Test that everything worked! 
        
        LS @DIAMONDS_ASSETS;

        '''

        st.code(setup_code_v2, language='sql')

    elif access==dont_know:
        setup_code_v3 = '''

        -- lists all roles granted to you 
        SHOW GRANTS TO YOUR_USERNAME;

        -- use this to check if you have access
        -- to create schemas on any databases
        SHOW GRANTS TO ROLE YOUR_ROLE;

        '''

        st.code(setup_code_v3, language='sql')

        st.write("Read more [here](https://docs.snowflake.com/en/sql-reference/sql/create-schema#access-control-requirements) about how to check if you have the create schema privilege.")

with tab2:
    st.write('If you are using Snowflake notebooks, you can copy and paste the below code directly into a new **Python** cell to check you are using the right warehouse and database. Your connection settings are already loaded in your active session -- no configuration necessary!')

    with st.expander("Expand to view the code"):
        code_connection ='''

        # Load Snowpark for Python
        from snowflake.snowpark import Session
        from snowflake.snowpark.version import VERSION  

        # Create a new Snowflake session object 
        # we'll use the session object throughout the demo
        session = get_active_session()
        session.sql_simplifier_enabled = True

        # Grab some environment variables to test our connection
        snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
        snowpark_version = VERSION

        # Print Current Environment Details
        print('Connection Established with the following parameters:')
        print('User                        : {}'.format(snowflake_environment[0][0]))
        print('Role                        : {}'.format(session.get_current_role()))
        print('Database                    : {}'.format(session.get_current_database()))
        print('Schema                      : {}'.format(session.get_current_schema()))
        print('Warehouse                   : {}'.format(session.get_current_warehouse()))
        print('Snowflake version           : {}'.format(snowflake_environment[0][1]))
        print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))
        '''

        st.code(code_connection)

    st.write('If you are using a Jupyter notebook on your local machine or a cloud provider, you will need to [configure your connection](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session) first. Below is a simple example to do this inside a notebook, but for production applications please use [a TOML file](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session#connect-by-using-the-connections-toml-file):')


    with st.expander("Expand to view the code"):
        code_connection2 ='''

        # Load Snowpark for Python
        from snowflake.snowpark import Session
        from snowflake.snowpark.version import VERSION  

        # setup connection
        connection_parameters = {
            "account": "YOUR_ACCOUNT",
            "user": "YOUR_USER",
            "password": "YOUR_PASSWORD",
            "role": "YOUR_ROLE",  # optional
            "warehouse": "YOUR_WAREHOUSE",  # optional
            "database": "YOUR_DATABASE",  # optional 
            "schema": "ML_HOL_SCHEMA",
            "client_session_keep_alive": True
        }


        # Create a new Snowflake session object 
        # we'll use the session object throughout the demo
        session = Session.builder.configs(connection_parameters).create()

        session.sql_simplifier_enabled = True

        # Grab some environment variables to test our connection
        snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
        snowpark_version = VERSION

        # Print Current Environment Details
        print('Connection Established with the following parameters:')
        print('User                        : {}'.format(snowflake_environment[0][0]))
        print('Role                        : {}'.format(session.get_current_role()))
        print('Database                    : {}'.format(session.get_current_database()))
        print('Schema                      : {}'.format(session.get_current_schema()))
        print('Warehouse                   : {}'.format(session.get_current_warehouse()))
        print('Snowflake version           : {}'.format(snowflake_environment[0][1]))
        print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))
        '''

        st.code(code_connection2)