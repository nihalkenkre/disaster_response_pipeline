import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, redirect, url_for, request, render_template

import sqlite3 as sql
import pandas as pd
import pickle

import sqlite3 as sql
import pandas as pd

import plotly
import plotly.express as px
import json

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__)

# Additional processing on the stopwords to remove punctuation since
# the punctuations are removed in the text before tokenizing
# and to add 'im' and 'un' since these words show up a lot in the features
# and they are not useful for training.

stop_words_regexed = [re.sub('[^\w\s]', '', sw)
                      for sw in stopwords.words('english')]
stop_words_regexed.append('im')
stop_words_regexed.append('u')
stop_words_regexed.append('un')

lemmatizer = WordNetLemmatizer()


def tokenize(text):
    '''
    Perform regex substitution (url, punctuation, numbers), tokenization, 
    lemmatization on the text and return the tokens

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


def read_pkl(input_pickle):
    '''
    Returns the pipeline model stored in the pkl file

    Parameters:
    input_pickle (str): Relative path to the pkl file

    Returns:
    trained_pipeline (sklearn.model_selection.GridSearchCV): Trained pipeline object

    '''

    trained_pipeline = None

    with open(input_pickle, 'rb') as input_file:
        trained_pipeline = pickle.load(input_file)

    return trained_pipeline


def read_sql(input_database):
    '''
    Returns the table from the database as a pandas dataframe

    Parameters:
    input_database (str): Relative path to the sqlite database

    Returns:
    df (pandas.DataFrame): Dataframe returned from the sqlite table
    '''

    conn = sql.connect(input_database)
    df = pd.read_sql('SELECT * from etl_data_output', conn)
    return df


def predict(query):
    '''
    Takes the query string and returns the classification array 

    Parameters:
    query (str): Query string to be classified

    Returns:
    predictions (np.ndarray): array of the categories that the query
                              was classified into
    '''

    # Read the model from the .pkl file
    model = read_pkl('./models/classifier.pkl')

    # Pass the query through a list to the predict method
    predictions = model.predict([query])

    return predictions


def get_category_labels():
    '''
    Return the names of the categories as a ndarray.
    This is got by filtering the column names of the dataframe returned from the database
    '''

    df = read_sql('data/etl_pipeline_database.db')

    return df.loc[:,
                  (df.columns != 'id') & (df.columns != 'message') &
                  (df.columns != 'original') & (df.columns != 'genre')
                  ].columns.values


def get_max_words_df(df):
    '''
    Returns dataframe with Top 10 Most appearing words in the message column along with the genre of the message

    Parameters:
    df (pandas.DataFrame): DataFrame returned from the sql table query

    Returns:
    df_max_appearance (pandas.DataFrame): Dataframe with the Top 10 most appearing words along with genre
    '''

    # Initialize a CountVectorizer with max_features=10. This will give us the 10 most appearing words.
    vect = CountVectorizer(max_features=10)
    mat = vect.fit_transform(df.message.tolist())

    # Create a dataframe with frequency matrix and the name of the words as columns
    df_max_appearance = pd.DataFrame(
        mat.todense(), columns=vect.get_feature_names_out())

    # Get a list of the column names (words) sorted by sum of all the genres from high to low
    sorted_col_names = pd.DataFrame(df_max_appearance.sum(), columns=['count']).sort_values(
        by='count', ascending=False).reset_index()['index'].tolist()

    # Inpute this string since the sum (above line) works on numeric_only and
    # because genre is not numeric it is left out
    sorted_col_names.insert(0, 'genre')

    # add the genre column to the dataframe
    df_max_appearance['genre'] = df.genre

    # Get the dataframe in the optimal format to be displayed as a stacked bar graph
    df_max_appearance = df_max_appearance.groupby('genre').sum().reset_index()[sorted_col_names].melt(
        id_vars='genre', var_name='word', value_name='count')

    return df_max_appearance


def get_combined_df(df):
    '''
    Create a dataframe with the categories grouped into macro categories along with genre

    Parameters:
    df (pandas.DataFrame): Dataframe grouped by 'genre' and summed

    Returns:
    df_combined (pandas.DataFrame): Dataframe with macro categories along with genre
    '''

    # Create a dataframe with the genre and macro categories as required.
    df_combined = pd.DataFrame(
        data={
            'genre': df.genre,
            'basics': df.request + df.food + df.water + df.clothing + df.shelter + df.electricity + df.money,
            'requests': df.request,
            'infrastructure': df.infrastructure_related + df.transport +
            df.buildings + df.hospitals + df.shops +
            df.aid_centers + df.buildings,
            'national_interest': df.search_and_rescue + df.security + df.military,
            'natural_elements': df.weather_related + df.floods + df.storm +
            df.fire + df.earthquake +
            df.cold + df.other_weather
        }
    )

    # Get a list of the column names (categories) sorted by sum of all the genres from high to low
    sorted_col_names = pd.DataFrame(df_combined.sum(numeric_only=True), columns=[
        'total']).sort_values(by='total', ascending=False).reset_index()['index'].tolist()

    # Inpute this string since the sum (above line) works on numeric_only and
    # because genre is not numeric it is left out
    sorted_col_names.insert(0, 'genre')

    # Get the dataframe in the optimal format to be displayed as a stacked bar graph
    df_combined = df_combined[sorted_col_names].melt(
        id_vars='genre', var_name='message_type', value_name='total')

    return df_combined


def get_data_for_plotly():
    '''
    Reads the database table as a pandas dataframe, and returns 2 dataframes
    to be displayed

    Returns:
    df_combined (pandas.DataFrame)  : datafram with categories into macro categories along with genre
    df_max_words (pandas.DataFrame) : dataframe with words appearing the most times along with genre 
    '''

    df = read_sql('data/etl_pipeline_database.db')
    df_groupby = df.groupby('genre', as_index=True).sum(
        numeric_only=True).drop(columns=['id']).reset_index()

    df_combined = get_combined_df(df_groupby)
    df_max_words = get_max_words_df(df)

    return df_combined, df_max_words


@app.route('/go.html', methods=['POST'])
def go_html():
    query = request.form['query']
    if len(query) == 0:
        return '<html>Please enter a valid query</html>'

    # get_category_labels is called to populate the 'table' with the category names
    # predict return the prediction of the model
    d = dict(zip(get_category_labels(), predict(query)[0]))

    return render_template('go.html', query=query, d=d)


@app.route('/')
def index_html():
    '''
    This function is called when the user requests the 'home page'.
    It get the dataframes for the graphs plots graphs, and passes the encoded json to the HTML template

    df_combined - DataFrame which groups the categories into macro categories, split by genre.

    Macro Categories are 
    Basic - Water, Shelter, Clothing, Food, Electricty, Money
    Natural Elements - Weather, Flood, Storm, Fire, Earthquake, Cold
    Infrastructure - Transport, Buildings, Hospitals, Shops, Aid Centers, Buildings
    National Interest - Search and Rescue, Security, Military Aid

    df_max_appearance - Dataframe which shows the words appearing the maximum of time in the messages, 
                        split by genre
    '''

    # get the dataframes for combined and the top 10 words
    df_combined, df_max_appearance = get_data_for_plotly()

    # plot the data using bargraph of plotly_express
    combined_fig = px.bar(df_combined, x='message_type', y='total', color='genre', labels={
        'message_type': 'Message Group', 'total': 'Total Messages'})

    # encode the graph data into json
    combined_graph = json.dumps(
        combined_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # plot the data using bargraph of plotly_express
    max_words_fig = px.bar(df_max_appearance, x='word',
                           y='count', color='genre', labels={'word': 'Top 10 most occuring words'})

    # encode the graph data into json
    max_words_graph = json.dumps(
        max_words_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', combined_graph=combined_graph, max_words_graph=max_words_graph)


@app.route('/favicon.ico')
def get_favicon():
    # Returning a number to prevent the error/warning on the console
    return b'0'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
