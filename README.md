# Udacity_ND_P2
# Udacity NanoDegree Project2 - Disaster Response Pipelines

### I. Summary

The project is to set up a ETL data pipeline that intakes twitter messages during disasters and the labeled categories of each message that are important to the response team, then build up a supervised machine learning model that predicts the labels according to the each message so that the model can classify new messages in case of a disaster. 

### II. Setup

There are three separate scripts that need to be executed for this project, while in the Udacity_ND_P2 directory:
1. To build up the ETL data pipeline
-`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. To build, train and save the model:
-`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. To run the API for interface:
-`cd app`
-`python run.py`

### III. Files Description

1. app
2. templates                 (template folder)
3. go.html                  (.html file for API)
4. master.html              (.html file for API)
5. run.py                    (.py file to run the API)
6. data
7. disaster_categories.csv   (.csv file for categories of messages)
8. disaster_messages.csv     (.csv file for raw messages)
9. DisasterResponse.db       (.db file for cleaned data in the database)
10. process_data.py           (.py file to build the ETL pipeline and generate the .db database file)
11. models
12. train_classifier.py       (.py file to build, train and save the model)
13. classifier.pkl            (.pkl file that saved the model, too large to load onto github)

### IV. Acknowledgements
The project opportunity is offered by Udacity Data Scientist NanoDegree Program and the date is sourced from Figure Eight.
