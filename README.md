# William-Hamant - Z23674646 - CAP4630 042 - Final Project
## Project Name: Pharmaceutical ​Interaction Classifier
A machine-learning system that predicts the severity of drug–drug interactions (Safe, Moderate, or Severe) based on medical text descriptions. This project includes dataset creation, model training using TF-IDF + Logistic Regression, evaluation metrics, and a fully functioning Flask web application for real-time predictions.

## Project Overview
Prescription interactions are a major cause of medical complications. This project demonstrates an AI-based solution that classifies the interaction severity between two medications using natural language descriptions.

The system:
- Builds a custom dataset of drug–drug interactions  
- Preprocesses medical text using TF-IDF vectorization  
- Trains a Logistic Regression classifier  
- Evaluates model performance  
- Deploys the model through a Flask web app

## How to run code:
The app can be deployed and show matrix tables by using the Windows Command Prompt. 
Once in the project file directory, navigate to the app folder by entering 'cd app' then start the Flask server by entering 'py app.py'
The Command Prompt will display an http: link that will take you to the website for the app interface.
To train the model, simply enter 'py -m src.train_model' while still in the project file directory.

## Presentation Recording
