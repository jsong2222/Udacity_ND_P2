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

-app
--templates                 (template folder)__
---go.html                  (.html file for API)__
---master.html              (.html file for API)__
--run.py                    (.py file to run the API)__
-data__
--disaster_categories.csv   (.csv file for categories of messages)__
--disaster_messages.csv     (.csv file for raw messages)__
--DisasterResponse.db       (.db file for cleaned data in the database)__
--process_data.py           (.py file to build the ETL pipeline and generate the .db database file)__
-models__
--train_classifier.py       (.py file to build, train and save the model)__
--classifier.pkl            (.pkl file that saved the model)__

### IV. Acknowledgements
The project opportunity is offered by Udacity Data Scientist NanoDegree Program and the date is sourced from Figure Eight.
