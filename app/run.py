# imports for script to run
import sys
from sqlalchemy import create_engine
import pandas as pd 
import numpy as np 
import re
import pickle
import json
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import nltk
nltk.download(['punkt','stopwords','wordnet','averaged_perceptron_tagger'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import plotly

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar


# functions and variables
app = Flask(__name__)

# additional stop words
add_stopwords = ['', ' ', 'say', 's', 'u', 'ap', 'afp', '...', 'n', '\\','http','bit','ly','like','know']

# stop_words
stop_words = ENGLISH_STOP_WORDS.union(add_stopwords)

# contractions dictionary to replace short versions with full versions
contractions_dict = {
  "i'm":"i am",
  "i'll":"i will",  
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "I would",
  "i'd've": "I would have",
  "i'll": "I will",
  "i'll've": "I will have",
  "i'm": "I am",
  "i've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

# compile
c_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# function to expand contractions
def expand_contractions(message, c_re=c_re):
    
    """
    Expands contractions according to pre-compiled dictionary.
    message - string, text where contractions have to be expanded.
    """
    
    def replace(match):
        return contractions_dict[match.group(0)]
    
    return c_re.sub(replace, message)


# a function to replace nltk pos tags with corresponding word_net pos tags (to use with WordNetLemmatizer)
def word_net_tags(nltk_tag):
    
    """
    Replaces ntlk pos tag with corresponding WordNet pos tag
    nltk_tag - nltk pos tag
    Returns a corresponding WordNet pos tag 
    """
    
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
    
# lemmatizing function
def lemmatize(tagged_text):
    
    """
    Lemmatizes tokens
    tagged_text - list of tokens and their nltk po tags
    Returns list of lemmatized tokens 
    """
    
    # list of lemmatized tokens
    lem_tokens = []
    
    # a for loop to populate the list
    for word,tag in tagged_text:
        
        word_net_tag = word_net_tags(tag)
        
        if word_net_tag is None:
            lem_tokens.append(WordNetLemmatizer().lemmatize(word))
            
        else:
            lem_tokens.append(WordNetLemmatizer().lemmatize(word, pos=word_net_tag))
            
            
    return lem_tokens


# text processing function for TfidfVectorizer
def process_text(text):
    
    """
    text - string
    Removes all non-alphabetica characters and stop words, lemmatizes the words in the text and tokenizes them
    Returns clean, lemmatized tokens
    """
    
    # decontract text
    text = expand_contractions(text.lower())
           
    # keep only letters and numbers
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
            
    # tokenize
    tokens = word_tokenize(text)
            
    # nltk pos tags
    tokens = nltk.pos_tag(tokens)
            
    # change nltk to wordnet pos tags and lemmatize
    tokens = lemmatize(tokens)
            
    # remove stop words and return clean tokens
    return [token for token in tokens if token not in stop_words]

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)
df.drop('child_alone', axis=1, inplace=True)

# load model
model = pickle.load(open('models/trained_classifier.pkl','rb'))

bigrams = pd.read_csv('bigrams.csv')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    message_class_counts = df.iloc[:,4:].sum().values
    message_class_names = df.iloc[:,4:].sum().index
    bigram_count = bigrams['count'].values
    bigrams_names = bigrams['bigram'].values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [

    # genre count
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

    # messages in each class
        {
            'data':[
                Bar(
                    x=message_class_counts,
                    y=message_class_names,
                    orientation='h'
                    )
            ],

            'layout':{
                'title':'Number of messages in each category',
                'yaxis':{
                    'title':'Category'
                },
                'xaxis':{
                    'title':'Number of messages'
                }
            }
        },

        # 10 most frequent bigrams
        {
            'data':[
                Bar(
                    x=bigram_count,
                    y=bigrams_names ,
                    orientation='h'   
                    )
            ],
            'layout':{
                'title':'10 most frequent bigrams',
                'yaxis':{
                    'title':'Bigram'
                },
                'xaxis':{
                    'title':'Count'
                }
            }    
        }     
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification probabilities for query
    class_proba = [round(proba,4) for proba in np.array(model.predict_proba([query]))[:,0,1]]
    classification_results = dict(zip(df.columns[4:], class_proba))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()