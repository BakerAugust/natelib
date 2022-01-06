'''
General utilities
'''

import pandas as pd
import snowflake.connector
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import os
from os.path import join
from dotenv import load_dotenv
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


def load_creds():
    '''
    Reads creds from .env file in the project root. 
    Make a .env file if you don't already have one and add this:
    SNOWFLAKE_USER=<your_username>
    SNOWFLAKE_PASSWORD=<your_password>
    '''
    dotenv_path = join('.env')
    load_dotenv(dotenv_path)
    pw = os.environ.get("SNOWFLAKE_PASSWORD")
    user = os.environ.get("SNOWFLAKE_USER")
    return(pw, user)


def sf_query(query, parse_dates=False):
    '''
    Return a dataframe from provided snowflake query
    '''
    pw, user = load_creds()

    account = 'indigoproduction.us-east-1'
    conn = snowflake.connector.connect(
            user=str(user),
            password=str(pw),
            account=str(account))
    
    if parse_dates:
        df = pd.read_sql(query, conn, parse_dates=parse_dates)
    else:
        df = pd.read_sql(query, conn)

    df.columns = [x.lower() for x in df.columns]
    
    return(df)


def df_to_snowlake(df, database_name, schema_name, role, table_name):
    '''
    Writes a dataframe to Snowflake 
    '''
    pw, user = load_creds()
    # Engine
    engine = create_engine(URL(
                            user = user,
                            password = pw,
                            account = 'indigoproduction.us-east-1',
                            role=role,
                            database = database_name,
                            schema = schema_name,
                            warehouse = 'DS_WH',
                            numpy = True))
    
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    return


def df_str2day_of_year(df, exclude_cols=[]):
    '''
    Converts all columns with date in the column name using pd.to_datetime
    
    Parameters
    ----------
    df : pd.DataFrame()
    Returns
    ----------
    df : pd.DataFrame()
    '''
    date_cols = [d for d in df.columns if 'date' in d]
    for col in date_cols:
        if col not in exclude_cols:
            df.loc[:,col]=pd.to_datetime(df[col]).dt.dayofyear
    
    return(df)


def df_day_of_year2datetime(df, exclude_cols=[]):
    '''
    Converts all columns with 'date' in the name to pd.datetime
    '''
    date_cols = [d for d in df.columns if 'date' in d]
    for col in date_cols:
        if col not in exclude_cols:
            df.loc[:,col]=pd.to_datetime(df[col]).dt.dayofyear

    return(df)


def read_rds(rds_path):
    '''Reads .RDS file into DataFrame'''
    pandas2ri.activate()
    base = importr('base')
    df = base.readRDS(rds_path)
    return df


def make_dir(path_to, dir_name, date=True):
    '''
    makes a new directory at specified path with todays 
    date, incrementing if the directory already exists. 
    '''
    
    if date:
        dir_path = path_to+dir_name+dt.date.today().strftime('%Y%m%d')
    else:
        dir_path = path+dir_name

    # Increment if exists
    while path.exists(dir_path):
        dir_path = dir_path + 'a'
    
    mkdir(dir_path)
    
    return dir_path + '/'


def make_readme(path_to, text=''):
    '''
    Adds a readme to specified path. Prompts user to input readme content. 
    '''
    
    print('readme notes:')
    user_input = input()
    readme = open(path_to + "readme.txt", "w") 
    readme.write(user_input + '\n' + text) 
    readme.close()


def f1_score(precision, recall):
    '''
    Calculates f1_score from precision and recall per
    https://en.wikipedia.org/wiki/F1_score#:~:text=The%20F1%20score%20is,Dice%20similarity%20coefficient%20(DSC).
    '''
    denominator = precision+recall
    numerator = precision*recall
    if denominator == 0 or numerator == 0:
        return 0
    else:
        f1_score = 2 * (numerator/denominator)
        return f1_score
