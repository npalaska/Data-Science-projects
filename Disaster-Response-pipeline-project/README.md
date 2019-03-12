# Disaster Response Pipeline Project
    
### Installations
I used python 3.7 for this project and the libraries used are the following

    sikit-learn ==0.19.1
    nltk == 3.2.5
    Flask==0.12.4
    numpy==1.12.1
    pandas==0.23.3
    plotly==2.0.15
    sqlalchemy==1.2.18

### Summary

This project is geared towards analyzing disaster data from Figure Eight. 
The Machine learning model is built that classifies disaster messages. 
The project contain real messages that were sent during disaster events. 
Machine learning model will enable the categorization of these events 
so that an appropriate disaster relief agency will be connected.


### File Descriptions

There are three main files in this project dedicated for their respective tasks:

1. **process_data.py**: This script loads the messages and categories datasets 
given by Figure Eight, merges them, cleans them and stores them into an SQLite
database.

2. **train_classifer.py**: This script loads back the data from the SQLite 
database, splits the database into the respective training and test sets, builds 
a text-processing and machine learning pipeline, trains and tunes the model 
using GridSearchCV, outputs the results on the test set, and finally exports the
final model as a pickle file.

3. **run.py**: This script is used by the flask web app and the visualizations 
are enabled using Plotly.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Screenshots
Here are some of the screen shots of the app that classifies the disaster event
messages and visualize it

![Alt text](classification.png?raw=true)

![Alt text](visualization.png?raw=true)

### Acknowledgements

This dataset is provided by Figure Eight and this work is done under Udacity
guidelines as a part of data science project.