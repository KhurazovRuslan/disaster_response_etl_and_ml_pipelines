<h1>Disaster response ETL and ML pipelines</h2>

The goal of the project is to build ETL and ML pipelines using the data provided by <a href='https://appen.com/'>Figure Eight (aquired by Appen)</a>, use those pipelines to create a classifier and deploy a web app that will be able to identify users' disaster messages or events in need of attention.


 <h2>Files:</h2>
    <ul>  
      <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/tree/main/app'>App folder</a>
        <ul>- <a href=''https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/tree/main/app/templates>templates folder</a> with .html files for web pages</ul>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/app/bigrams.csv'>bigrams.csv</a> file with data for web page visualization</ul>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/app/run.py'>run.py</a> script to deploy web app</ul>
</ul>
    <ul>   
      <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/tree/main/data'>Data folder</a>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/data/DisasterResponse.db'>DisasterResponse.db</a> database file where the data is stored after ETL pipeline is run</ul>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/data/EDA%20and%20ETL.ipynb'> EDA and ETL</a> jupyter notebool with exploritory data analysis and steps for ETL pipeline</ul>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/data/categories.csv'>categories.csv</a> is initial file containing all labels for disaster messages to train the model on</ul>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/data/messages.csv'>messages.csv</a> is initial file with disaster messages to train the model on</ul>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/data/process_data.py'>process_data.py</a> is ETL pipeline script</ul>
</ul>
    <ul>   
      <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/tree/main/model'>Model folder</a>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/model/ML.ipynb'>ML</a> jupyter notebook with description of the model's training, tuning and evaluation processes</ul>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/model/train_classifier.py'>train_classifier.py</a> is a machine learning pipeline script</ul>
        <ul>- <a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/model/trained_classifier.pkl'>trained_classifier.pkl</a> is already trained and tuned classifier to deploy with web app</ul>
</ul>  
    <ul><a href='https://github.com/KhurazovRuslan/disaster_response_etl_and_ml_pipelines/blob/main/README.md'>Readme</a> file</ul>      

  <h2>Techologies used:</h2>
      <ul>- python 3.7.8</ul>
      <ul>- pandas 1.1.0</ul>
      <ul>- numpy 1.18.5</ul>
      <ul>- matplotlib 3.3.0</ul>
      <ul>- seaborn 0.10.1</ul>
      <ul>- re 2.2.1</ul>
      <ul>- nltk 3.5</ul>
      <ul>- wordcloud 1.8.1</ul>
      <ul>- scikit-learn 0.23.2</ul>
      <ul>- gensim 3.8.3</ul>
      <ul>- sqlalchemy 1.3.22</ul>
      <ul>- json 2.0.9</ul>
      <ul>- plotly 4.9.0</ul>
      <ul>- flask 1.1.2</ul> 

  <h2>Instructions:</h2>
  <ul>
    Run the following commands in the project's root directory to set up your database and model.
    <ul> -To run ETL pipeline that cleans data and stores in database: python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db</ul>
    <ul>- To run ML pipeline that trains classifier and saves it: python model/train_classifier.py data/DisasterResponse.db model/classifier.pkl</ul>
    <ul>- Type the following command to run your web app: python app/run.py</ul>
    <ul>- Go to http://0.0.0.0:3001/ (in my case it was https://view6914b2f4-3001.udacity-student-workspaces.com/). In order to classify your message, type it in the box and press 'Classify Message', wait for the result. If there is a high possibily of the message belonging to one of the classes that category will be highlighted in red (at least 80% probability), if it is likely to belong to one of the classes that category will be highlighted in orange (between 50 and 80% probability), if it doesn't belong to a class that category will be highlighted in green (less than 50% probability)</ul>
</ul>

  <h2>Thanks to:</h2>
  <ul>https://towardsdatascience.com/using-word2vec-to-analyze-news-headlines-and-predict-article-success-cdeda5f14751 and https://medium.com/@robert.salgado/multiclass-text-classification-from-start-to-finish-f616a8642538 for explaining basics of NLP for multiclass problems</ul>
  <ul>https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/#:~:text=There%20are%20two%20groups%20of,%2Dspecificity%20and%20precision%2Drecall. for explaining different metrics for classification problems</ul>
  <ul>And all of data science and programming community for being very open and desire to share your ideas!</ul>

