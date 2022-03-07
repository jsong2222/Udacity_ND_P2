import sys
import re
import pandas as pd
import pickle
import joblib
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','omw-1.4'])

from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def load_data(database_filepath):
    """
        Load data from Database and split the X and y

        Args:
            database_filepath: path to .db file
        Return:
            X: DataFrame, preditor messages
            y: DataFrame, target labels
            category_names: list, name of labels
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X,y, category_names


def tokenize(text):
    """
        Tokenize the text:
            change url to placeholder
            normalize the word token
            remove stopwords
            Lemmatize the token
        Args:
            text: string, messages
        Return:
            tokens: list of string

    """


    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize and tokenize and remove punctuation
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # Lemmatize
    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens


def build_model():
    """
        Build the pipeline and parameters for model

        Args: 
            None
        Return:
            model: GridSearchCV model object

    """
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3]
    }

    model = GridSearchCV(pipeline, param_grid=parameters,n_jobs=3, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
        Evaluate the model for all categories

        Args:
            model: GridSearchCV model object
            X_test: DataFrame, test predictor
            y_test: DataFrame, test target
            category_name: list of string, name of labels
        Return:
            class_report: report of classification

    """
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)
    return class_report    


def save_model(model, model_filepath):
    """
        Save model to pkl file

        Args:
            model: GridSearchCV model object
            model_filepath: path to save model
        Return:
            None
    """
#     joblib.dump(model.best_estimator_, model_filepath, compress = 1)
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()