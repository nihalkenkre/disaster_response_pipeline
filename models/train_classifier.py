import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

import argparse
import sqlite3 as sql
import pandas as pd
import pickle

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Additional processing on the stopwords to remove punctuation since
# the punctuations are removed in the text before tokenizing
# and to add 'im' and 'un' since these words show up a lot in the features
# and they are not useful for training.

stop_words_regexed = [re.sub('[^\w\s]', '', sw)
                      for sw in stopwords.words('english')]
stop_words_regexed.append('im')
stop_words_regexed.append('un')

lemmatizer = WordNetLemmatizer()


def read_data_from_db(input_database):
    '''
    Reads data from the input_database and returns X, y arrays. 

    Parameters:
    input_database (str): Relative path to the sqlite database.

    Returns:
    X (numpy.ndarray)   : array of the text of messages
    y (pandas.DataFrame): Dataframe of the categories values
    category_labels     : Names for the category
    '''

    # Connect to the database using the sqlite module
    conn = sql.connect(input_database)

    # Read the table as a pandas dataframe
    table_name = 'etl_data_output'
    df = pd.read_sql('SELECT * FROM {}'.format(table_name), conn)

    # Extract the raw message values from the dataframe
    X = df.message.values

    # Extract the columns with the category values
    y = df.loc[:, (df.columns != 'id') & (df.columns != 'message') & (
        df.columns != 'original') & (df.columns != 'genre')]

    # labels for the cateory columns
    category_labels = y.columns.values

    return X, y, category_labels


def tokenize(text):
    '''
    Perform regex substitution (url, punctuation, numbers), tokenization, lemmatization on the text and return the tokens

    We have performed an additional process on the stopwords.word('english') list. Removed the punctuations
    and added 'im' and 'un' to the list to better clean up this dataset.

    Parameters:
    text (str): The text to perform the tokenization on

    Returns:
    tokens (list): cleaned up tokens
    '''

    regex_url = '\w?http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    regex_url_spaced = '\w?http[s]?\s[a-zA-Z]+.[a-zA-Z]+\s\w+'
    regex_punct = '[^\w\s]'
    regex_numbers = '[0-9]'

    text = re.sub(regex_url, '', text)
    text = re.sub(regex_url_spaced, '', text)
    text = re.sub(regex_punct, '', text)
    text = re.sub(regex_numbers, '', text)

    # tokenize the incoming text
    all_tokens = word_tokenize(text)

    # Add tokens to passed_tokens which are not a part of stop_words_regexed list
    passed_tokens = [at for at in all_tokens if at not in stop_words_regexed]

    tokens = []

    # We do not lemmatize 'us' it gives a 'u' which is out of context in any context.
    for pt in passed_tokens:
        if pt != 'us':
            tokens.append(lemmatizer.lemmatize(pt))
        else:
            tokens.append(pt)

    return tokens


def get_pipeline():
    '''
    Builds a pipeline object and returns it

    Returns:
    grid_search_cv (sklearn.model_selection.GridSearchCV): The pipeline object
    '''

    # List of dicts for hyperparameters to perform the Grid Search on.
    # Attempting to perform this GridSearchCV took a long time so had to cut it short multiple times.
    param_grid = [
        {'vect__max_features': [5000, 10000, 20000, 30000]},
        {'vect__n_gram_range': [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3)]},
        {'vect__min_df' : [0.0001, 0.001, 0.01]},
        {'clf__estimator__n_estimators': [50, 100, 200]}
    ]

    pipeline = Pipeline(
        [
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1), n_jobs=-1))
        ]
    )

    # Create a gridsearch object with pipeline and hyper parameter grid
    grid_search_cv = GridSearchCV(pipeline, param_grid, verbose=1)

    return pipeline


def display_results(y_pred, y_test, category_labels, trained_pipeline):
    '''
    Prints a dataframe containing the precision_scores, f1_scores, recall_scores,
                      accuracy_scores for each of the 36 catagories, 
                      based on the inputs y_pred and y_test

    Parameters:
    y_pred (numpy.ndarray)      : array of predicted outputs
    y_test (pandas.DataFrame)   : Test data from the dataset
    catagory_labels             : Labels to be used as columns names in the output dataframe
    trained_pipeline            : The trained pipeline object
    '''

    y_test_array = y_test.values

    precision_scores = []
    f1_scores = []
    recall_scores = []
    accuracy_scores = []

    # We go through the y_test and y_pred values and get the required metrics for each of the 36 categories 
    # and present them in a dataframe
    for i in range(y_test_array.shape[1]):
        precision_scores.append(precision_score(
            y_test_array[:, i], y_pred[:, i], average='weighted'))
        f1_scores.append(f1_score(
            y_test_array[:, i], y_pred[:, i], average='weighted'))
        recall_scores.append(recall_score(
            y_test_array[:, i], y_pred[:, i], average='weighted'))
        accuracy_scores.append(accuracy_score(
            y_test_array[:, i], y_pred[:, i]))

    df = pd.DataFrame([precision_scores, f1_scores, recall_scores,
                      accuracy_scores], columns=category_labels, index=['precision', 'f1', 'recall', 'accuracy'])

    print(df.transpose())#, 'Best params', trained_pipeline.best_params_)


def write_pickle(trained_pipeline, output_pickle):
    '''
    Writes the trained_pipeline object to a pkl file to the path given by the output_pickle

    Parameters:
    trained_pipeline (sklearn.model_selection.GridSearchCV) : The pipeline object
    output_pickle (str)                                     : The relative path to the pkl file
    '''

    with open(output_pickle, 'wb') as output_file:
        pickle.dump(trained_pipeline, output_file)


def read_pickle(input_pickle):
    '''
    Read the pickle file which contains the trained pipeline object and return the pipeline object

    Parameters:
    input_pickle (str): Relative path to the .pkl file

    Returns:
    trained_pipeline (sklearn.model_selection.GridSearchCV): The pipeline object
    '''

    trained_pipeline = None

    with open(input_pickle, 'rb') as input_file:
        trained_pipeline = pickle.load(input_file)

    return trained_pipeline


def main(args):
    print('Extracting Data from {} ...'.format(args.input_database))

    # Get the data from the database.
    # X - The message texts
    # y - the category classfications
    X, y, category_labels = read_data_from_db(args.input_database)

    # Split the data between training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    trained_pipeline = None

    print('Building pipeline ...')
    trained_pipeline = get_pipeline()

    print('Fitting pipeline to training data...')
    trained_pipeline.fit(X_train, y_train)

    print("Writing out pipeline to Pickle file {} ...".format(args.pickle_file))
    write_pickle(trained_pipeline, args.pickle_file)

    print('Testing pipeline ...')
    y_pred = trained_pipeline.predict(X_test)

    print('Displaying results ...')
    display_results(y_pred, y_test, category_labels, trained_pipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates a machine learning model which is trained to the input data and exports the trained model as a Pickle file')

    parser.add_argument('input_database', type=str,
                        help='Relative path to the database file')

    parser.add_argument('pickle_file', type=str,
                        help='Relative path to the pickle file')

    args = parser.parse_args()

    main(args)
