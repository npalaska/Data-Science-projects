import sys
import pickle
from joblib import dump, load
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection  import GridSearchCV

nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    """
    Function to load cleaned data
    
    Arguments:
        database_filepath: path to sql database
    Output:
        X: feature DataFrame
        Y: label DataFrame
        category_names: labled categories which can be used for visualizing
        model validations
    """
    
    engine = create_engine('sqlite:///'+database_filepath)
    
    # Read the cleaned disaster response sql table 
    df = pd.read_sql_table('disasterResponseCleaned', engine)
    
    df.loc[df.related == 2, 'related'] = 1
    
    X = df.message.values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text: list of English text messages
    Output:
        clean_tokens: tokenized text, cleaned for building ML model
    """
    # Text normalizing
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Tokenize the text and remove the english stopwords such as and, the, etc. 
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]

    # Lemmatize the extracted tokens
    lemmatizer = WordNetLemmatizer()
    final_tokens = []
    for token in tokens:
        tok = lemmatizer.lemmatize(token).strip()
        final_tokens.append(tok)

    return final_tokens


def build_model():
    '''
    Build the machine learning pipeline. 
    The ML pipeline do the following:
        1) Feature extraction using the CountVectorizer which Convert a
        collection of text documents to a matrix of token counts
        2) Transform the extracted token matrix to a normalized tf-idf
        representation
        3) Building a machine learning model
    '''
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
    
    parameters = {
        'tfidf__smooth_idf':[True],
        'clf__estimator__dual': [False],
        'clf__estimator__C':  [1]
    }

    # create grid search object
    cv = GridSearchCV(model, param_grid=parameters, n_jobs=-1, cv=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out
    model performance in terms of accuracy, precision, recall and f score
    
    Arguments:
        model: ML Model that we have built
        X_test: test samples
        Y_test: test labels
        category_names: label names
    """
    Y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        accuracy = '%25s accuracy : %.2f' \
                   %(category_names[i],
                     accuracy_score(Y_test[:,i], Y_pred[:,i]))
        print(accuracy)
    with open('result.log','w') as file:
        print(classification_report(Y_test, Y_pred,
                                    target_names=category_names))
        file.write(classification_report(Y_test, Y_pred,
                                         target_names=category_names))
    file.close()


def save_model(model, model_filepath):
    """
    Function to save the trained ML model
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model: GridSearchCV or an ML object 
        model_filepath: destination path to save .pkl file
    
    """
    #with open(model_filepath, 'wb') as file:
    #    pickle.dump(model, file)
    #dump(model, open(model_filepath, 'wb'))
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
