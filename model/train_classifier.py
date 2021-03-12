# QUICK NOTE!
# This script was tested in Udacity workspace IDE, however,
# udacity IDE doesn't seem to have xgboost library installed, so
# I used scikit learn's Decision tree classifier instead (only in IDE)
# if you run it udacity workspace IDE, uncomment lines reffering to Decision tree and comment 
# lines reffering xgboost

# THE REST OF THE CODE STAYS THE SAME!



# imports for script to run
import sys
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt','wordnet','stopwords','averaged_perceptron_tagger','omw'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
#from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import recall_score, f1_score, brier_score_loss, roc_auc_score, classification_report


import pickle


# functions and variables for text preprocessing

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

           

# a function to load the data from the database
def load_data(database_file_path):
    """
    Loads the data from SQLite database
    database_file_path - string, path to sqlite database (.db file)
    Returns: X - dataframe with features to train and test on,
             y - dataframe with multi-class(35) binary labels(0,1)
    """
    engine = create_engine(f'sqlite:///{database_file_path}')
    df = pd.read_sql('SELECT * FROM disaster_messages', engine)
    df.drop('child_alone', axis=1, inplace=True)
    X = df['message']
    y = df.iloc[:,4:]

    return X, y


# a function to build a machine learning model 
def build_model():
    """
    Builds a machine learning pipeline and sets the search for the best parameters for that model.
    Returns machine learning model with sets of parameters to search within
    """ 

    # variable for a pipeline
    pipeline = Pipeline(steps=[
      ('features',TfidfVectorizer(tokenizer=process_text, ngram_range=(1,3), max_df=0.95, min_df=2)),
      #('clf',MultiOutputClassifier(XGBClassifier(random_state=42, max_depth=6, n_estimators=1000, eta=0.1)))])
      ('clf',MultiOutputClassifier(DecisionTreeClassifier(random_state=42)))])
        


    # set of parameters to search for better result
    # just one parameter to save time. for full version refer to ML notebook
    params = {'features__ngram_range':[(1,2),(1,3)]}
    

    # initialize model, 2-fold fitting to save time
    model = GridSearchCV(estimator=pipeline, param_grid=params, cv=2, scoring='recall_weighted', verbose=5)


    return model 

# function to evaluate probabilities
def evaluate_proba(y_test,y_proba,metric):
    """
    Evaluates predicted probabilities
    y_test - true labels
    y_proba - predicted probabilities
    metric - string, one of 2 metrics, 'brier_score_loss' or 'roc_auc_score'
    Returns the mean score of each predicted probabilities' labels
    """
    
    # list of scores of each label
    scores = []
    
    # a for loop to populate the list
    for i,category in enumerate(list(y_test.columns)):
        
        # if roc_auc_score chosen
        if metric=='roc_auc_score':
            scores.append(roc_auc_score(y_test[category],y_proba[i][:,1]))
            
        elif metric=='brier_score_loss':
            scores.append(brier_score_loss(y_test[category],y_proba[i][:,1]))
            
    return np.mean(scores)                          

# a function for model evaluation
def evaluate_model(model,X_test,y_test):
    """
    Evaluates a machine learning model with scikit-learn classification report and recall_score
    model - trained machine learning model to evaluate
    X_test - features for model testing
    y_test - labels for model testing
    Prints out a classification report for each class and overall recall_score
    """

    # predictions
    y_pred = np.array(model.predict(X_test))
    y_proba = np.array(model.predict_proba(X_test))

    # a for loop to print out classification report for each class
    for i,column in enumerate(list(y_test.columns)):
        print(f'Classification report for {column} column:')
        print(classification_report(y_test[column],y_pred[:,i]))
        print(' ')

    print(f"Overall recall score is {recall_score(y_test,y_pred, average='weighted')}") 
    print(f"Overall f1_score score is {f1_score(y_test,y_pred, average='weighted')}")
    print(f"Brier score loss is {evaluate_proba(y_test,y_proba,'brier_score_loss')}")
    print(f"ROC AUC score is {evaluate_proba(y_test,y_proba,'roc_auc_score')}")


# a function to save a machine learning model
def save_model(model,model_file_path):
    """
    Saves model into .pkl file for later use
    model - machine learning model to save
    model_file_path - string, a file path to .pkl file
    """
    pickle.dump(model, open(model_file_path,'wb'))

# a main function to run the script
def main():

    if len(sys.argv) == 3:

        database_file_path, model_file_path = sys.argv[1:]


        print(f'Loading the data from SQLite database...')
        X, y = load_data(database_file_path)

        # train - test - split
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        print(' ')

        print('Building the model...')
        model = build_model()
        print(' ')

        print('Training the model...')
        model.fit(X_train,y_train)
        print(' ')

        print('Evaluating the model...')
        evaluate_model(model,X_test,y_test)
        print(' ')

        print('Saving the model...')
        save_model(model,model_file_path)
        print(' ')

        print(f'All done! Trained model is successfully stored in {model_file_path} file')

    else:

        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl') 


if __name__=='__main__':
    main()   