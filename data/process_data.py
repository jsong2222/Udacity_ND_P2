import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
        Load two files into dataframe and join on id

        Args: 
            messages_filepath (relative path)
            categories_filpath (relative path)
        Return:
            df_combined (DataFrame)
    """
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    df_combined = df_messages.merge(df_categories,how='inner', on='id')
    return df_combined

def clean_data(df):
    """
        Clean DataFrame:
            Expand categories
            Generate column names
            Convert all categories to 1 or 0
            Drop original category column
            Remove Duplicates
        Args:
            df DataFrame
        Return:
            df DataFrame
    """
    split_category = df['categories'].str.split(";", expand=True)
    #generate column names
    row = split_category.iloc[0,:]
    category_colnames = list(row.apply(lambda x: x[:-2]))
    split_category.columns = category_colnames

    #convert to 1 and 0
    for column in split_category:
        split_category[column] = split_category[column].str[-1]
        split_category[column] = split_category[column].astype(int)
    
    split_category.replace(2, 1, inplace=True)

    #drop and concatenate categories
    df = df.drop('categories', axis=1)
    df = pd.concat([df, split_category], axis=1)

    #remove duplicates
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset='id')

    return df

def save_data(df, database_filename):
    """
        Load dataframe to database file

        Args:
            df Dataframe
            database_filename .pb file location
        Return:
            None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)  


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