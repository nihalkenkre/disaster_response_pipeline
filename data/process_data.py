import argparse
import pandas as pd
import sqlite3 as sql

import re

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Additional processing on the stopwords to remove punctuation since
# the punctuations are removed in the text before tokenizing
# and to add 'im' and 'un' since these words show up a lot in the features
# and they are not useful for training.

stop_words_regexed = [re.sub('[^\w\s]+', '', sw)
                      for sw in stopwords.words('english')]
stop_words_regexed.append('im')
stop_words_regexed.append('un')

lemmatizer = WordNetLemmatizer()

def extract(messages_csv, categories_csv):
    '''
    Extract data from the input CSV files

    Paramters:
    messages_csv (str)  : Path to the csv file containing the messages
    categories_csv (str): Path to the csv file containing the categories

    Returns:
    df_m: Pandas DataFrame created from the messages from messages_csv file
    df_c: Pandas DataFrame created from the categories from the categories_csv file
    '''

    df_m = pd.read_csv(messages_csv)
    df_c = pd.read_csv(categories_csv)

    return df_m, df_c


def cleanup_message(msg):
    '''
    NOT USED CURRENTLY 
    since it does remove the . and ' punctuation marks. 

    Run a regex substitution for URLs, punctuation and numbers, since they are not important for the classification

    Parameters:
    msg (str): The message to be cleaned up

    Returns:
    msg (str): The cleaned up message
    '''

    regex_url = '\w?http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    regex_url_spaced = '\w?http[s]?\s[a-zA-Z]+.[a-zA-Z]+\s\w+'

    regex_punct = '(\(|\)|\.|\,|\?|-|!|:|\'|"|\/|;|\[|\])+'
    regex_numbers = '[0-9]+'

    msg = re.sub(regex_url, '', msg, count=100, flags=re.IGNORECASE)
    msg = re.sub(regex_url_spaced, '', msg, count=100, flags=re.IGNORECASE)
    msg = re.sub(regex_punct, '', msg, count=100, flags=re.IGNORECASE)
    # msg = msg.replace("'", "").replace(
    # ".", "").replace(",", "").replace("?", "").replace("-", "")\
    # .replace("(", "").replace(")", "").replace("\"", "").replace("&", "")\
    # .replace("[", "").replace("]", "")
    #msg = re.sub(regex_numbers, '', msg, re.IGNORECASE)

    # tokenize the incoming text
    all_tokens = word_tokenize(msg)

    # Add tokens to passed_tokens which are not a part of stop_words_regexed list
    passed_tokens = [at.lower() for at in all_tokens if at not in stop_words_regexed]

    tokens = []

    # We do not lemmatize 'us' it gives a 'u' which is out of context in any context.
    for pt in passed_tokens:
        if pt != 'us':
            tokens.append(lemmatizer.lemmatize(pt))
        else:
            tokens.append(pt)

    msg = " ".join(tokens)
    
    return msg


def cleanup_categories(df_c):
    '''
    Converts the string based categorical values into integer based values for ease of math operations

    Parameters:
    df_c (pandas.DataFrame): String based catagorical dataframe

    Returns:
    df_c_sep_id (pandas.DataFrame): Integer based categorical dataframe with categories as column names
    '''

    # The df_c.categories is split and with expand=True, splits into 36 columns, 1 for each category
    # The columns are labeled 0-35
    df_c_sep = df_c.categories.str.split(';', expand=True)

    # The column values currently are e.g. related-1 request-0 food-1 shelter-1
    # We get the string value before the '-' which is the column name
    col_names = df_c_sep.iloc[0].apply(lambda x: x.split('-')[0]).values

    # Column names assigned to the dataframe
    df_c_sep.columns = col_names

    # The column values currently are e.g. related-1 request-0 food-1 shelter-1
    # We get the integer value after the '-' which is the category value
    df_c_sep = df_c_sep.apply(
        lambda x: x.apply(lambda y: y.split('-')[1]))

    # Since the values are 1 or 0, convert the dataframe to integer for ease of operations
    df_c_sep = df_c_sep.astype('int')

    # There are some values in the df_c.related which are 2
    # Closer inspection of the messages with the index revealed incoherence in the messages in the df_m
    # Also no other category is set to 1
    # So we will replace the 2s in the related column with 0s
    df_c_sep.replace(2, 0, inplace=True)

    # Concatenate the newly created category values with the original category dataframe.
    # Drop the original 'categories', since it is not longer required
    # Now the integer values of the categories are aligned the 'id' column
    df_c_sep_id = pd.concat(
        [df_c, df_c_sep], axis=1).drop(columns=['categories'])

    return df_c_sep_id


def transform(df_m, df_c):
    '''
    Transform the incoming data as required.
    Cleans up the df_c and merges with df_m.

    Parameters:
    df_m (pandas.DataFrame): messages Dataframe
    df_c (pandas.DataFrame): categories Dataframe

    Returns:
    pandas.DataFrame: Merged Dataframe
    '''

    # Cleanup the message column
    df_m.message = df_m.message.apply(lambda msg: cleanup_message(msg))

    # Cleanup categories
    df_c = cleanup_categories(df_c)

    # Merge the categories dataframe with the messages dataframe on 'id'
    # This dataframe has the messages aligned correctly with the category values
    df = pd.merge(df_m, df_c, on='id')

    # We drop duplicate rows, after merging
    # since dropping duplicates before merging would give dataframes with different number of rows,
    # leading to loss of data after merge
    df.drop_duplicates(inplace=True)

    return df


def load(df, output_database_path):
    '''
    Loads the incoming dataframe to a sqlite database

    Parameters:
    df (pandas.DataFrame)       : The dataframe to save
    output_database_path (str)  : The relative path of the database file
    '''

    conn = sql.connect(output_database_path)

    table_name = 'etl_data_output'
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    conn.close()


def main(args):
    print('Extracting data from {} and {} ...'.format(
        args.input_messages, args.input_categories))
    df_m, df_c = extract(args.input_messages, args.input_categories)

    print('Transforming data ...')
    df = transform(df_m, df_c)

    print('Loading data to {} ...'.format(args.output_database_path))
    load(df, args.output_database_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract and Tranform the messages and categories data and Load it into a SQLite DB')
    parser.add_argument('input_messages', type=str,
                        help='Relative path to the messages csv file')
    parser.add_argument('input_categories', type=str,
                        help='Relative path to the categories csv file')
    parser.add_argument('output_database_path', type=str,
                        help='Relative path to the database file')

    args = parser.parse_args()

    main(args)
