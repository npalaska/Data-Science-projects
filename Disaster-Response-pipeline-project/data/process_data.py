import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """
    Function to load message and categories data
    
    Arguments:
        messages_filepath: path to messages csv file
        categories_filepath: path to categories csv file
    Output:
        df: Loaded data as Pandas DataFrame
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
        
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Function to clean the loaded dataframe
    
    Arguments:
        df: raw DataFrame object
    Outputs:
        df: clean DataFrame object
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = list(map(lambda x: x[:len(x)-2], row))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # Check if the category has more than 2 unique values
        # If so pre-process it so it contains only 2 unique values
        if len(categories[column].unique())>2:
            print("column with more than 2 unique value: ", column)
            categories[column] = categories[column].apply(lambda x:1 if x>1 else 0)
        
        # Check if the category has only one value
        # If so remove that category since it does not add any valuable information about the data
        if len(categories[column].unique())==1:
            print("column with only one unique value: ", column)
            categories = categories.drop(column, axis=1)
        
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)
    
    # check number of duplicates and if duplicates present drop them
    if (len(df[df.duplicated()]) > 0):
        df = df.drop_duplicates()
    
    assert len(df[df.duplicated()]) == 0
    return df
    
    
def save_data(df, database_filename):
    """
    Function to save the cleaned DataFrame to a specified database file
    
    Arguments:
        df: Clean DataFrame object
        database_filename: database file (.db) destination path
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save dataframe to database, relace if already exists 
    df.to_sql('disasterResponseCleaned', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
