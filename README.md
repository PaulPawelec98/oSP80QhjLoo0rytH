Apziva Project 2: Bank Term Deposit Subscription Prediction
Overview

This project aims to predict whether clients will subscribe to a bank's term deposit based on various personal and campaign-related attributes. By analyzing the dataset, we identify key factors influencing client decisions and develop a predictive model to enhance marketing strategies.

The primary dataset used in this project is term-deposit-marketing-2020.csv. This dataset contains information on direct marketing campaigns (phone calls) aiming to predict if the client will subscribe to a term deposit.

Each row represents an individual client's information, with the following columns:

    age: Age of the client (numeric)
    job: Type of job (categorical)
    marital: Marital status (categorical)
    education: Level of education (categorical)
    default: Has credit in default? (binary: "yes","no")
    balance: Average yearly balance in euros (numeric)
    housing: Has housing loan? (binary: "yes","no")
    loan: Has personal loan? (binary: "yes","no")
    contact: Contact communication type (categorical)
    day: Last contact day of the month (numeric)
    month: Last contact month of the year (categorical)
    duration: Last contact duration in seconds (numeric)
    campaign: Number of contacts performed during this campaign for this client (numeric)
    y: Has the client subscribed to a term deposit? (binary: "yes","no")

Project Structure

    Project.ipynb: Jupyter Notebook containing the main analysis and modeling workflow.
    analysis.py: Python script for data preprocessing and exploratory data analysis.
    analysis2.py: Additional Python script for supplementary analysis.
    term-deposit-marketing-2020.csv: Dataset file containing client information and campaign outcomes.

Installation

To run the analysis and models locally, ensure you have the following Python packages installed:

    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn

You can install these packages using pip:

pip install pandas numpy scikit-learn matplotlib seaborn

Acknowledgments

Apziva for providing the dataset and project framework.​ The open-source community for their invaluable tools and libraries.​