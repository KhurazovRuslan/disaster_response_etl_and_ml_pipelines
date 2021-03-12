# imports for script to run
import sys
import pandas as pd
from sqlalchemy import create_engine


# a function to read the data
def read_data(m_file_path,c_file_path):
    """
    Reads the data from message.csv and categories.csv
    and returns merged dataframe
    m_file_path - string, a path to messages.csv
    c_file_path - string, a path to categories.csv
    """
    
    # read the data
    messages = pd.read_csv(m_file_path)
    categories = pd.read_csv(c_file_path)
    
    # merge
    df = pd.merge(messages,categories, on='id')
    
    return df


# a function to clean the data
def clean_data(df):
    """
    Cleans the data and returns a dataframe
    df - dataframe to clean
    """
    
    # list of categories to use as column names 
    categories_cols = [names.split('-')[0] for names in df['categories'][0].split(';')]
    
    # creating 36 individual category columns
    for i in range(len(categories_cols)):
        df[categories_cols[i]] = [int(row.split(';')[i].split('-')[1]) for row in df['categories']]
        
    # labels 0 and 2 in 'related' class are similar (refer to notebook)
    # change 2s into 0s to make it more simple
    df['related'] = df['related'].map({0:0,1:1,2:0})
    
    # drop 'categories' column
    df.drop('categories', axis=1, inplace=True)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df    



# a function to save the data into database
def save_data(df,database):
    """
    Saves the data as disaster_messages table in SQLite database
    df - dataframe to save
    database - string, database name
    """
    
    # creating a connection to database
    engine = create_engine(f'sqlite:///{database}')
    
    # save the data
    df.to_sql(name='disaster_messages', con=engine, index=False)



# put everything together
def main():
    """
    Puts everything in one flow
    """
    if len(sys.argv)==4:

        # files path
        m_file_path,c_file_path,database = sys.argv[1:]

        # first, read the data
        print('Reading the data...')
        df = read_data(m_file_path,c_file_path)
        print('OK!')
        print(' ')
        
        # clean it
        print('Cleaning the data...')
        df = clean_data(df)
        print('OK!')
        print(' ')
        
        # save it
        print('Saving data...')
        save_data(df,database)
        print(' ')
        
        # when it's done
        print(f'Cleaned data is stored in {database[:-3]} database')  

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()