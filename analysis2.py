# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:34:38 2025

@author: Paul
"""

# %% [1] Import Packages

# Standard Library
import os
# import random  # Uncomment if needed

# Core Data Handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical Modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# General ML Frameworks
import lazypredict
from lazypredict.Supervised import LazyClassifier

# Transformations & Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer

# Modeling & Training
# from sklearn.utils import shuffle
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression  # for ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron

# dimension reduction
from sklearn.decomposition import TruncatedSVD

# optimization
from sklearn.model_selection import GridSearchCV

# Evaluation Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import classification_report

# Ensemble
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

# Hyper Parameter Tuning
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll.base import scope
from hyperopt import space_eval
import optuna

# %% [2] Background
'''
Background:

We are a small startup focusing mainly on providing machine learning solutions
 in the European banking market. We work on a variety of problems including
 fraud detection, sentiment classification and customer intention prediction
 and classification.

We are interested in developing a robust machine learning system that leverages
 information coming from call center data.

Ultimately, we are looking for ways to improve the success rate for calls made
 to customers for any product that our clients offer. Towards this goal we are
 working on designing an ever evolving machine learning product that offers
 high success outcomes while offering interpretability for our clients to make
 informed decisions.

Data Description:

The data comes from direct marketing efforts of a European banking institution.
 The marketing campaign involves making a phone call to a customer, often
 multiple times to ensure a product subscription, in this case a term deposit.
 Term deposits are usually short-term deposits with maturities ranging from one
 month to a few years. The customer must understand when buying a term deposit
 that they can withdraw their funds only after the term ends. All customer
 information that might reveal personal information is removed due to privacy
 concerns.

Attributes:

age : age of customer (numeric)

job : type of job (categorical)

marital : marital status (categorical)

education (categorical)

default: has credit in default? (binary)

balance: average yearly balance, in euros (numeric)

housing: has a housing loan? (binary)

loan: has personal loan? (binary)

contact: contact communication type (categorical)

day: last contact day of the month (numeric)

month: last contact month of year (categorical)

duration: last contact duration, in seconds (numeric)

campaign: number of contacts performed during this campaign and for this
 client (numeric, includes last contact)

Output (desired target):

y - has the client subscribed to a term deposit? (binary)
Download Data:

https://drive.google.com/file/d/1EW-XMnGfxn-qzGtGPa3v_C63Yqj2aGf7

Goal(s):

Predict if the customer will subscribe (yes/no) to a term deposit (variable y)
Success Metric(s):

Hit %81 or above accuracy by evaluating with 5-fold cross validation and
 reporting the average performance score.

Current Challenges:

We are also interested in finding customers who are more likely to buy the
 investment product. Determine the segment(s) of customers our client should
 prioritize.

What makes the customers buy? Tell us which feature we should be focusing
 more on.
'''
# %% [3] Setup Environment and Data

# Seed settings
# seed = random.randint(1000, 9999)
# print(seed)

seed = 6932
np.random.seed(seed)

# Show all rows and columns
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 1000)  # Adjust width for long outputs

# set directory
path = r'E:\My Stuff\Projects\Apziva\4132339939'
os.chdir(path)

# Custom Class
# from HyperOptObj import HyperOptObj

df = pd.read_csv('term-deposit-marketing-2020.csv')

# %% [4] Exploratory

# Setup -----------------------------------------------------------------------
df.describe()

# Check Categorical Numbers

_cols_categorical = [
    col for col, dtype in df.dtypes.items() if dtype == 'O' and col != 'y'
    ]

_cols_numerical = [
    col for col, dtype in df.dtypes.items() if not dtype == 'O' and col != 'y'
    ]
# -----------------------------------------------------------------------------

# Corr Matrix -----------------------------------------------------------------
df_temp = df.copy()
df_temp['y'] = df['y'].replace({'yes': 1, 'no': 0})
sns.heatmap(df_temp.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

'''
duration is the most prominant feature.
    - balance has a weak effect...
'''

# -----------------------------------------------------------------------------

# Bar Plots -------------------------------------------------------------------
'''
Bar plots for each categorical variable to show the count of each class
'''

# Create subplots
fig, axes = plt.subplots(5, 2, figsize=(16, 16))

# Flatten the axes array to match with categorical columns
axes_flat = axes.flatten()

# Iterate through the flattened axes and categorical columns
for ax, column in zip(axes_flat, _cols_categorical):
    _data = dict(df[column].value_counts())
    _classes = _data.keys()
    _values = _data.values()

    # Create a unique color for each class
    colors = plt.cm.tab10(np.arange(len(_classes)))

    # Group bars by class and create a bar plot
    bars = ax.bar(_classes, _values, color=colors)

    ax.set_title(f'Count of {column.upper()}')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count of Category')

    # Add labels on each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval),
                ha='center', va='bottom')  # Position label above the bar

    # Rotate x-axis labels
    ax.set_xticklabels(_classes, rotation=45, ha='right')

# Remove empty subplots if any (if _cols_categorical has fewer than 8 columns)
for i in range(len(_cols_categorical), len(axes_flat)):
    fig.delaxes(axes_flat[i])

fig.suptitle("Bar Plots of Categorical", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()

'''
Not all even between groups for each plot.

Seems fairly representitive of the population, pretty normal standard, mostly
middle class people.

Odd thing to note, huge amount of people in May?
'''
# -----------------------------------------------------------------------------

# Grouped Bar Plots -----------------------------------------------------------
'''
Grouped bar plots for each categorical variable to show the count of each class
'''

# Create subplots
fig, axes = plt.subplots(4, 2, figsize=(16, 16))

# Flatten the axes array to match with categorical columns
axes_flat = axes.flatten()

# Iterate through the flattened axes and categorical columns
for ax, column in zip(axes_flat, _cols_categorical):
    _group = df.groupby([column, 'y'])[column].count()
    _group.name = 'value'
    _data = _group.reset_index()
    _pivot = _data.pivot_table(
        index=_data.columns[0],
        columns=_data.columns[1],
        values='value'
        )
    _pivot.plot(kind='bar', stacked=False, ax=ax)

    ax.set_title(f'Count of {column.upper()}')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count of Category')

fig.suptitle("Bar Plots of Categorical", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()

'''
The amount of people that have subscribed to term deposits is extremely small
 when compared accross various classes in each plot. Nothing really discernable
 or anything to note.
'''
# -----------------------------------------------------------------------------

# Bar Plot Y=1 ----------------------------------------------------------------
# Create subplots
fig, axes = plt.subplots(4, 2, figsize=(16, 16))

# Flatten the axes array to match with categorical columns
axes_flat = axes.flatten()

# Iterate through the flattened axes and categorical columns
for ax, column in zip(axes_flat, _cols_categorical):
    _group = df.groupby([column, 'y'])[column].count()
    _group.name = 'value'
    _data = _group.reset_index()
    _data = _data[_data['y'] == 'yes']
    _pivot = _data.pivot_table(
        index=_data.columns[0],
        columns=_data.columns[1],
        values='value'
        )
    _pivot.plot(kind='bar', stacked=False, ax=ax, color='orange')

    ax.set_title(f'Count of {column.upper()}')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count of Category')

fig.suptitle("Bar Plots of Categorical (Y=1)", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()

'''
Managment, blue color, technicians, admin have term deposits
married have the most
secondary education
more have no defaults
housing is even
no loans
celluar
may???
'''
# -----------------------------------------------------------------------------

# Violin Plots for Numerical Columns ------------------------------------------

fig_violin, axes = plt.subplots(3, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    try:

        _col = _cols_numerical[i]

        cutoff = {
            'balance': 10000,
            'duration': 2000,
            'campaign': 10
            }

        if _col in cutoff.keys():

            rows = df[_col] < int(cutoff[_col])
            _ydata = df.loc[rows, _col]
            _xdata = df.loc[rows, 'y']

        else:
            _ydata = df.loc[:, _col]
            _xdata = df['y']

        sns.violinplot(x=_xdata, y=_ydata, ax=ax)
        ax.set_title(f"{_col}")

    except Exception:
        ax.axis('off')

plt.tight_layout()
fig_violin.suptitle("Violin Plots")
plt.show()

'''
Nothing polarizing except for duration! Seems like longer durations were able
to turn clients onto term deposits.

- seems like lower incomes wouldn't sign onto term deposits? (balance)
'''
# -----------------------------------------------------------------------------

# Mean of y -------------------------------------------------------------------
y_df = df['y'].replace({'yes': 1, 'no': 0})
y_df.mean()
y_df.value_counts()

'''
Super small mean!! 0.0724 Very small sample of people who have a term deposit.
'''
# -----------------------------------------------------------------------------

# %% [5A] Logistic Regressions
'''
- run single regressions
- review results
'''

# Single Logistic Regressions Function ----------------------------------------
'''
Function to run each indepent variable on the dependent.
'''


def run_single_regressions(y_df, x_df):
    results = []
    pred = []
    models = []

    for col in x_df.columns:
        X = x_df[col]
        y = df['y'].replace({'yes': 1, 'no': 0})

        if X.dtype == 'O':
            dummies = pd.get_dummies(X, drop_first=True)
            X = pd.concat([dummies], axis=1)

        X = sm.add_constant(X)

        # Fit the regression model
        model = sm.Logit(y, X).fit()
        models.append(model)

        # Predict
        pred.append(model.predict(X))

        # Collect the results
        results.append({
            'Variable': col,
            'Coefficient': model.params[1],
            'Intercept': model.params[0],
            'P-Value': model.pvalues[1],
            # 'R-Squared': model.rsquared
        })
    return results, pred, models


# -----------------------------------------------------------------------------

# Return results --------------------------------------------------------------
results, pred, models = run_single_regressions(
    df['y'], df[df.columns[0:-1]]
    )

dfresults = summary_col(models, stars=True, float_format='%0.4f')
dfresults

# dfresults.tables[0].to_csv("result_tables.csv")

'''
Results tend to point at individuals who would have low risk tolerance would
also have term deposits.

Unit increases in duration leads to a higher probability of conversion, with
statistical significance. The impact is fairly small per unit, but duration is
measured in minutes.
    - average duration is 254.82 seconds, or around 4 minutes
    - is this because they are actually converting them because they are
    convincing them? Or are they already interested in term deposits, and thus
    since they are interested or wanting a term deposit the conversation goes
    on longer?

Seems like the focus should be targeting individuals that are likely already
considering term deposits. Those whou have low risk tolerances...
    - unemployed
    - no debts, no loans, older (middle age)

I think it's unlikely for people to convince individuals to change their risk
tolerance.
'''
# -----------------------------------------------------------------------------

# %% [5B] Multivariate Logistic Regressions
'''
More risk averse individuals I think would typically be:
    - job: low skill, low education, unemployed, etc...
'''

# Variables
reg_results = {}
data = df.copy()
data['y'] = data['y'].replace({'yes': 1, 'no': 0})

# duration ~ job --------------------------------------------------------------
reg1 = "duration~job"
reg_results[reg1] = smf.ols(reg1, data=data).fit()
# -----------------------------------------------------------------------------

# duration ~ 'Kitchen Sink' ---------------------------------------------------
cols = list(df.columns)
cols.remove('duration')
reg2 = f"duration~{('+').join(cols)}"
reg_results[reg2] = smf.ols(reg2, data=data).fit()
# -----------------------------------------------------------------------------

# y ~ duration ----------------------------------------------------------------
reg3 = "y~duration"
reg_results[reg3] = smf.logit(reg3, data=data).fit()
# -----------------------------------------------------------------------------

# y ~ balance -----------------------------------------------------------------
# data_temp = data.copy()
# data_temp['balance'] = np.log(data['balance'] - + min(data['balance']) + 1)
reg4 = 'y~balance'
reg_results[reg4] = smf.logit(reg4, data=data).fit()
# -----------------------------------------------------------------------------

# y ~ 'Kitchen Sink'  ---------------------------------------------------------
cols = list(df.columns)
cols.remove('y')
reg5 = f"y~{('+').join(cols)}"
reg_results[reg5] = smf.logit(reg5, data=data).fit()
# -----------------------------------------------------------------------------

# duration ~ y ----------------------------------------------------------------
reg6 = 'duration~y'
reg_results[reg6] = smf.ols(reg6, data=df).fit()
# -----------------------------------------------------------------------------

summary_col(reg_results[reg6], stars=True, float_format='%0.4f')

'''
"duration~job"
    - blue collar*** +
    - self-employed** +
    - unemployed*** +

f"duration~{('+').join(cols)} - everything
    - blue collar, services, self-employed continue to be statistically and
    economical significance
    - services*** +, housemaid*** +, entrepreneur** +, retired** +,
    student*** -, all of these are statisically significant now, but weren't
    before.
    - They are going to have much longer conversations be more likely to
    subscribe to a term deposit.
    - duration increases by 482 seconds when y == 1?!

"y~duration"
    - higher duration gives a positive effect on a client subscribing ***.

"y~balance"
    - higher account balance gives a negligible positve effect ***.

f"y~{('+').join(cols)}"
    - same story
    - retired and higher education seems to have the strongest effects of
    indicating they would buy term deposits.
        job[T.retired]         0.3090**
                               (0.1384)
        education[T.tertiary]  0.3149***
                               (0.0978)
        default[T.yes]         0.2990*
                               (0.1728)
        balance                0.0000**
                               (0.0000)

Seems like anything that indicates lower risk tolerance means they are more
likely to take on a term deposit.

Could further test by random sampling the data into smaller pieces
'''

# %% [6A] Setup Data for First Layer
'''
hot-encoding
standardization

After some thought, it seems like the variables duration (mostly this),
 day, month, and campaign could be leaking the answer to our model...

For instance, in the multivariate logistic regressions, we can that y has a
huge effect on duration. So essentially, people subscribing to term deposits
would have longer calls. This becomes a variable that is not so much as
explaining/predicting the outcome but kind of just showing it.

day, month, campaign might be in a similar boat, so for the inital model
implementation, I will focus primarily on the segmentation of people and less
about the actuals calls happening.
    - you can see how these variables are the most important on the
    feature importance graph.

'''

# Setup X and y
X = df[df.columns[0:-1]]
y = df['y']
y = y.replace({'yes': 1, 'no': 0})

# Drop Columns
X = X.drop(columns=['duration', 'month', 'day', 'campaign'])


# Train and Test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=seed,
    stratify=y  # Ensures equal class distribution
)

# encoding

# Create Pipelines for numerical columns and categorical columns
num_pipeline = make_pipeline(
    StandardScaler()
)


cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
)


# Create Preprocessing variable to apply pipelines
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=int)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)

# Create Training Set and Test Set
X_train_t = preprocessing.fit_transform(X_train)
X_test_t = preprocessing.fit_transform(X_test)

X_train_t_df = pd.DataFrame(
    X_train_t,
    columns=preprocessing.get_feature_names_out(),
    index=X_train.index)


X_test_t_df = pd.DataFrame(
    X_test_t,
    columns=preprocessing.get_feature_names_out(),
    index=X_test.index)

# Create Full Set for Later... To Trim
X_trim = preprocessing.fit_transform(X)

X_trim_df = pd.DataFrame(
    X_trim,
    columns=preprocessing.get_feature_names_out(),
    index=X.index)


# %% Feature Importance

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=seed)
rf.fit(X_train_t_df, y_train)

# Get feature importance
importance = rf.feature_importances_

# Plot feature importance
plt.figure(figsize=(16, 24))

# Create a horizontal bar plot
plt.barh([x for x in range(len(importance))], importance, color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature Index")
plt.title("Feature Importance from Random Forest")
plt.yticks(range(len(importance)), labels=X_train_t_df.columns, fontsize=10)
plt.show()

# %% [7A] Model Selection - First Layer

# # drop duration
# X_train_t_df = X_train_t_df.drop(columns=['pipeline-1__duration'])
# X_test_t_df = X_test_t_df.drop(columns=['pipeline-1__duration'])

# LazyPredict
clf = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    custom_metric=None,
    predictions=True,
    random_state=seed
    )

lazymodels, predictions = clf.fit(
    X_train_t_df,
    X_test_t_df,
    y_train,
    y_test
)

lazymodels = pd.DataFrame(lazymodels)

# Figure for cms
fig_cm, axes = plt.subplots(9, 3, figsize=(24, 48))

cm_results = {}

# Get predictions for each model
for i, ax in enumerate(axes.flat):
    try:
        # get model and pred data
        model_name = predictions.keys()[i]
        y_pred = predictions[model_name]

        # Generate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_results[model_name] = cm

        # Plot Confusion Matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Pred: 0', 'Pred: 1'],
            yticklabels=['True: 0', 'True: 1'],
            ax=ax,
            cbar=False  # Remove the colorbar
            )

        ax.set_title(f'Confusion Matrix for {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    except Exception:
        ax.axis('off')

fig_cm.suptitle("Confusion Matrix Heatmap: LazyPredict", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

'''
                               Accuracy  Balanced Accuracy  ROC AUC  F1 Score  Time Taken
Model
NearestCentroid                    0.59               0.60     0.60      0.68        4.09
GaussianNB                         0.88               0.55     0.55      0.88        0.12
QuadraticDiscriminantAnalysis      0.53               0.52     0.52      0.63        0.15
ExtraTreesClassifier               0.90               0.52     0.52      0.88        4.40
RandomForestClassifier             0.92               0.52     0.52      0.89        4.31
ExtraTreeClassifier                0.88               0.52     0.52      0.87        0.13
DecisionTreeClassifier             0.86               0.52     0.52      0.87        0.27
LabelSpreading                     0.91               0.51     0.51      0.89      116.01
BaggingClassifier                  0.92               0.51     0.51      0.89        1.50
LabelPropagation                   0.91               0.51     0.51      0.89       84.86
PassiveAggressiveClassifier        0.84               0.51     0.51      0.85        0.11
KNeighborsClassifier               0.92               0.51     0.51      0.89        1.98
AdaBoostClassifier                 0.93               0.51     0.51      0.89        1.59
XGBClassifier                      0.93               0.51     0.51      0.89        1.08
LGBMClassifier                     0.93               0.50     0.50      0.89        0.61
BernoulliNB                        0.93               0.50     0.50      0.89        0.14
LogisticRegression                 0.93               0.50     0.50      0.89        1.23
LinearDiscriminantAnalysis         0.93               0.50     0.50      0.89        0.74
DummyClassifier                    0.93               0.50     0.50      0.89        0.09
CalibratedClassifierCV             0.93               0.50     0.50      0.89       20.75
RidgeClassifier                    0.93               0.50     0.50      0.89        0.12
RidgeClassifierCV                  0.93               0.50     0.50      0.89        0.17
SGDClassifier                      0.93               0.50     0.50      0.89        0.22
SVC                                0.93               0.50     0.50      0.89       43.18
LinearSVC                          0.93               0.50     0.50      0.89        5.67
Perceptron                         0.86               0.50     0.50      0.86        0.11
'''
# -----------------------------------------------------------------------------

# Closer Look at CM Results ---------------------------------------------------
cm_ratios_df = pd.DataFrame(columns=['tp', 'tn', 'fp', 'fn'])

for key, value in cm_results.items():

    tn, fp, fn, tp = value.flatten()

    cm_ratios_df.loc[key] = {
            'tp': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tn': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fp': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fn': fn / (fn + tp) if (fn + tp) > 0 else 0
        }

cm_ratios_df.sort_values(by='tp', ascending=False)

'''
NearestCentriod has the best balanced scores, and is one of the only models
that is doing better than random.

                                tp   tn   fp   fn
NearestCentroid               0.62 0.58 0.42 0.38
QuadraticDiscriminantAnalysis 0.51 0.53 0.47 0.49
GaussianNB                    0.17 0.94 0.06 0.83
PassiveAggressiveClassifier   0.13 0.89 0.11 0.87
DecisionTreeClassifier        0.11 0.92 0.08 0.89
ExtraTreeClassifier           0.09 0.94 0.06 0.91
Perceptron                    0.08 0.92 0.08 0.92
ExtraTreesClassifier          0.08 0.96 0.04 0.92
LabelPropagation              0.05 0.98 0.02 0.95
RandomForestClassifier        0.05 0.99 0.01 0.95
LabelSpreading                0.04 0.98 0.02 0.96
BaggingClassifier             0.03 0.99 0.01 0.97
KNeighborsClassifier          0.02 0.99 0.01 0.98
XGBClassifier                 0.01 1.00 0.00 0.99
AdaBoostClassifier            0.01 1.00 0.00 0.99
LGBMClassifier                0.01 1.00 0.00 0.99
BernoulliNB                   0.01 1.00 0.00 0.99
LogisticRegression            0.00 1.00 0.00 1.00
LinearDiscriminantAnalysis    0.00 1.00 0.00 1.00
DummyClassifier               0.00 1.00 0.00 1.00
CalibratedClassifierCV        0.00 1.00 0.00 1.00
RidgeClassifier               0.00 1.00 0.00 1.00
RidgeClassifierCV             0.00 1.00 0.00 1.00
SGDClassifier                 0.00 1.00 0.00 1.00
SVC                           0.00 1.00 0.00 1.00
LinearSVC                     0.00 1.00 0.00 1.00
'''
# -----------------------------------------------------------------------------

# %% [8] Model Implementation - First Layer
'''
I can't get probabilities from NearestCentriod, so I use Gausian to trim our
data, it had the 2nd best balanced accuracy of our predictors.
'''

# Variables
fitted_models = {}

# NearestCentroid -------------------------------------------------------------
# param_grid = {
#     'metric': ['euclidean', 'manhattan'],
#     'shrink_threshold': [None, 0.001, 0.01, 0.2, 0.5]
# }

# model = GridSearchCV(
#     NearestCentroid(),
#     param_grid,
#     cv=5,
#     scoring='balanced_accuracy',
#     n_jobs=-1
#     )

# model.fit(X_train_t_df, y_train)

# fitted_models['NearestCentroid'] = model
# -----------------------------------------------------------------------------

# QDA -------------------------------------------------------------------------
# param_grid = {
#     'reg_param': [None, 0.001, 0.01, 0.2, 0.5],  # Similar to shrinkage
#     'tol': [1e-4, 1e-3, 1e-2]  # Tolerance for singular value decomposition
# }

# # Initialize GridSearchCV with QDA
# model = GridSearchCV(
#     QuadraticDiscriminantAnalysis(),
#     param_grid,
#     cv=5,
#     scoring='balanced_accuracy',
#     n_jobs=-1
# )

# # Fit the model
# model.fit(X_train_t_df, y_train)

# # Store the fitted model
# fitted_models['QDA'] = model
# -----------------------------------------------------------------------------

# Define hyperparameter grid --------------------------------------------------
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# Initialize GridSearchCV with GaussianNB
model = GridSearchCV(
    GaussianNB(),
    param_grid,
    cv=5,
    scoring='balanced_accuracy',  # Handles class imbalance better
    n_jobs=-1
)

# Fit the model
model.fit(X_train_t_df, y_train)

# Store the fitted model
fitted_models['GaussianNB'] = model
# -----------------------------------------------------------------------------

# Metrics
y_pred = model.predict(X_test_t_df)

# Compute metrics
print(classification_report(y_test, y_pred))

'''
              precision    recall  f1-score   support

           0       0.94      0.94      0.94     18552
           1       0.17      0.17      0.17      1448

    accuracy                           0.88     20000
   macro avg       0.55      0.55      0.55     20000
weighted avg       0.88      0.88      0.88     20000
'''

# Trim

y_pred = fitted_models['GaussianNB'].predict_proba(X_trim_df)
df_trimmed = df[(y_pred[:, 0] < 0.99)]

'''
Let's remove all cases where the probability of the customer not subscribing
to a term deposit is higher than 0.99, this drops around 10,000 calls.
    - These people are probably not worth calling.
    - Only drops arounds 200 people that would have been worth calling as
    a draw back.

'''

# %% [9] Model Selection - Second Layer

'''
Set up the newly trimmed dataset for ML, and then test some models again.
    - Now that we have a group of callers we'd like to be calling, let put
    back our variables of duration, day, month, campaign, etc...
'''

# Setup X and y
X = df_trimmed[df.columns[0:-1]]
y = df_trimmed['y']
y = y.replace({'yes': 1, 'no': 0})

# Train and Test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=seed,
    stratify=y  # Ensures equal class distribution
)

# encoding

# Create Pipelines for numerical columns and categorical columns
num_pipeline = make_pipeline(
    StandardScaler()
)


cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
)


# Create Preprocessing variable to apply pipelines
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=int)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)

# Create Training Set and Test Set
X_train_t = preprocessing.fit_transform(X_train)
X_test_t = preprocessing.fit_transform(X_test)

X_train_t_df = pd.DataFrame(
    X_train_t,
    columns=preprocessing.get_feature_names_out(),
    index=X_train.index)


X_test_t_df = pd.DataFrame(
    X_test_t,
    columns=preprocessing.get_feature_names_out(),
    index=X_test.index)

# Create Full Set for Later... To Trim
X_trim = preprocessing.fit_transform(X)

X_trim_df = pd.DataFrame(
    X_trim,
    columns=preprocessing.get_feature_names_out(),
    index=X.index)

# LazyPredict
clf = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    custom_metric=None,
    predictions=True,
    random_state=seed
    )

lazymodels, predictions = clf.fit(X_train_t_df,
                                  X_test_t_df,
                                  y_train,
                                  y_test
                                  )

lazymodels = pd.DataFrame(lazymodels)

# Figure for cms
fig_cm, axes = plt.subplots(9, 3, figsize=(24, 48))

cm_results = {}

# Get predictions for each model
for i, ax in enumerate(axes.flat):
    try:
        # get model and pred data
        model_name = predictions.keys()[i]
        y_pred = predictions[model_name]

        # Generate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_results[model_name] = cm

        # Plot Confusion Matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Pred: 0', 'Pred: 1'],
            yticklabels=['True: 0', 'True: 1'],
            ax=ax,
            cbar=False  # Remove the colorbar
            )

        ax.set_title(f'Confusion Matrix for {model_name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    except Exception:
        ax.axis('off')

fig_cm.suptitle("Confusion Matrix Heatmap: LazyPredict", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

'''
'''
# -----------------------------------------------------------------------------

# Closer Look at CM Results ---------------------------------------------------
cm_ratios_df = pd.DataFrame(columns=['tp', 'tn', 'fp', 'fn'])

for key, value in cm_results.items():

    tn, fp, fn, tp = value.flatten()

    cm_ratios_df.loc[key] = {
            'tp': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tn': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fp': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fn': fn / (fn + tp) if (fn + tp) > 0 else 0
        }

cm_ratios_df.sort_values(by='tp', ascending=False)

'''

GaussianNB                    1.00 0.04 0.96 0.00
QuadraticDiscriminantAnalysis 0.98 0.05 0.95 0.02
NearestCentroid               0.76 0.89 0.11 0.24
Perceptron                    0.50 0.94 0.06 0.50
DecisionTreeClassifier        0.44 0.95 0.05 0.56
XGBClassifier                 0.44 0.97 0.03 0.56
LGBMClassifier                0.43 0.98 0.02 0.57
LinearDiscriminantAnalysis    0.43 0.97 0.03 0.57
PassiveAggressiveClassifier   0.39 0.84 0.16 0.61
AdaBoostClassifier            0.37 0.97 0.03 0.63
BaggingClassifier             0.36 0.98 0.02 0.64
SGDClassifier                 0.33 0.98 0.02 0.67
LabelPropagation              0.31 0.96 0.04 0.69
LabelSpreading                0.31 0.96 0.04 0.69
LogisticRegression            0.31 0.98 0.02 0.69
ExtraTreeClassifier           0.31 0.95 0.05 0.69
CalibratedClassifierCV        0.30 0.98 0.02 0.70
RandomForestClassifier        0.29 0.98 0.02 0.71
LinearSVC                     0.24 0.99 0.01 0.76
SVC                           0.23 0.99 0.01 0.77
ExtraTreesClassifier          0.22 0.98 0.02 0.78
KNeighborsClassifier          0.19 0.99 0.01 0.81
RidgeClassifier               0.17 0.99 0.01 0.83
RidgeClassifierCV             0.17 0.99 0.01 0.83
BernoulliNB                   0.12 0.99 0.01 0.88
DummyClassifier               0.00 1.00 0.00 1.00

Highest balanced accuracies and ROC AUC
- NearestCentriod
- Perceptron
- XGBClassifier

    Model	Balanced Accuracy	ROC AUC
NearestCentroid	0.8228180687844189	0.8228180687844189
Perceptron	0.7187339649165132	0.7187339649165131
XGBClassifier	0.7072609061764937	0.7072609061764937

'''
# -----------------------------------------------------------------------------

# %% [10] Model Implementation - Second Layer
'''
- NearestCentriod
- Perceptron
- XGBClassifier
'''

fitted_models = {}

# NearestCentroid -------------------------------------------------------------
param_grid = {
    'metric': ['euclidean', 'manhattan'],
    'shrink_threshold': [None, 0.001, 0.01, 0.2, 0.5]
}

model = GridSearchCV(
    NearestCentroid(),
    param_grid,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1
    )

model.fit(X_train_t_df, y_train)

fitted_models['NearestCentroid'] = model
# -----------------------------------------------------------------------------

# XGBClassifier -------------------------------------------------------------
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.8, 1.0]
}

model = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid_xgb,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1
)

model.fit(X_train_t_df, y_train)
fitted_models['XGBClassifier'] = model
# -----------------------------------------------------------------------------

# Perceptron -------------------------------------------------------------
param_grid_perceptron = {
    'penalty': [None, 'l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'max_iter': [1000, 2000, 5000]
}

model = GridSearchCV(
    Perceptron(),
    param_grid_perceptron,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1
)

model.fit(X_train_t_df, y_train)
fitted_models['Perceptron'] = model
# -----------------------------------------------------------------------------

# Gather Accuracy Scores ------------------------------------------------------


def get_accuracy_score(x):
    model = fitted_models[x]
    y_pred = model.predict(X_test_t_df)
    y_pred = (y_pred >= 0.5).astype(int)
    result = accuracy_score(y_test, y_pred)
    return x, result


def get_confusion_matricies(x):
    model = fitted_models[x]
    y_pred = model.predict(X_test_t_df)
    y_pred = (y_pred >= 0.5).astype(int)
    result = confusion_matrix(y_test, y_pred)
    return x, result


accuracy = dict(
    map(lambda x: get_accuracy_score(x), fitted_models)
    )

confusion_matricies = dict(
    map(lambda x: get_confusion_matricies(x), fitted_models)
    )

# -----------------------------------------------------------------------------

# Confusion Plots -------------------------------------------------------------
fig_cm2, axes = plt.subplots(3, 2, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    try:
        cm = confusion_matricies[list(confusion_matricies.keys())[i]]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred: 0', 'Pred: 1'],
                    yticklabels=['True: 0', 'True: 1'],
                    ax=ax
                    )
        ax.set_title(f'{list(confusion_matricies.keys())[i]}')
    except Exception:
        ax.axis('off')

fig_cm2.suptitle("Confusion Matrix Heatmap: ML", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# Confusion Ratios ------------------------------------------------------------
cmr = pd.DataFrame(columns=['tp', 'tn', 'fp', 'fn'])

for key, value in confusion_matricies.items():

    tn, fp, fn, tp = value.flatten()

    cmr.loc[key] = {
            'tp': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tn': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fp': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fn': fn / (fn + tp) if (fn + tp) > 0 else 0
        }

cmr.sort_values(by='tp', ascending=False)
# -----------------------------------------------------------------------------

# %% [11] Further Optimization

'''
Can't really optimize nearest centriod much further, but XGB looks to be
promising.


- "objective": "multi:softmax"
    for multiclassification

- "eval_metric": "aucpr"
    for imbalanced datasets, should be good for imbalanced datasets.

        How AUC-PR Helps with Imbalanced Data

        Less Impact from Majority Class
            AUC-PR ignores True Negatives, which means a large majority class
            doesn’t artificially boost the score.

        Focuses on Precision & Recall
            If the minority class is small, AUC-PR ensures that the model
            isn't just predicting the majority class correctly while
            ignoring the minority.

'''


# # Define the Optuna objective function
# def objective(trial):
#     # Define hyperparameter search space
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 300),

#         "max_depth": trial.suggest_int("max_depth", 3, 15),

#         "learning_rate": trial.suggest_float(
#             "learning_rate", 0.01, 0.3, log=True
#             ),

#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "use_label_encoder": False,
#         "objective": "multi:softmax",
#         "num_class": 2,
#         "eval_metric": "aucpr",
#         "seed": seed,
#     }

#     # Train the model with these parameters
#     model = XGBClassifier(**params, random_state=seed)

#     scores = cross_val_score(
#         model, X_train_t_df, y_train, cv=5, scoring="average_precision"
#      )

#     return scores.mean()  # Maximize accuracy


# # Run the Optuna optimization
# study = optuna.create_study(direction="maximize")  # Maximize accuracy
# study.optimize(objective, n_trials=50, n_jobs=-1)

# # Print the best parameters
# print("Best accuracy:", study.best_value)
# print("Best parameters:", study.best_params)

best_params = {
  'n_estimators': 297,
  'max_depth': 15,
  'learning_rate': 0.08890903365180643,
  'subsample': 0.9576066506798475,
  'colsample_bytree': 0.6404825013379966,
  'gamma': 3.1695839150861618,
  'reg_alpha': 0.011568052561632403,
  'reg_lambda': 4.584734978300553,
  'min_child_weight': 3
  }

# Train the best model with the found parameters
# best_params = study.best_params
best_model = XGBClassifier(**best_params, random_state=seed)
best_model.fit(X_train_t_df, y_train)

fitted_models['Optuna:XGB'] = best_model

# Scores ----------------------------------------------------------------------
accuracy = dict(
    map(lambda x: get_accuracy_score(x), fitted_models)
    )

confusion_matricies = dict(
    map(lambda x: get_confusion_matricies(x), fitted_models)
    )
# -----------------------------------------------------------------------------

# Confusion Plots -------------------------------------------------------------
fig_cm2, axes = plt.subplots(3, 2, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    try:
        cm = confusion_matricies[list(confusion_matricies.keys())[i]]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred: 0', 'Pred: 1'],
                    yticklabels=['True: 0', 'True: 1'],
                    ax=ax
                    )
        ax.set_title(f'{list(confusion_matricies.keys())[i]}')
    except Exception:
        ax.axis('off')

fig_cm2.suptitle("Confusion Matrix Heatmap: ML", fontsize=16, y=1.00)
plt.tight_layout()
plt.show()
# -----------------------------------------------------------------------------

# Confusion Ratios ------------------------------------------------------------
cmr = pd.DataFrame(columns=['tp', 'tn', 'fp', 'fn'])

for key, value in confusion_matricies.items():

    tn, fp, fn, tp = value.flatten()

    cmr.loc[key] = {
            'tp': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tn': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fp': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'fn': fn / (fn + tp) if (fn + tp) > 0 else 0
        }

cmr.sort_values(by='tp', ascending=False)
# -----------------------------------------------------------------------------

'''
Optimization didn't seem to improve much...

                  tp   tn   fp   fn
NearestCentroid 0.67 0.90 0.10 0.33
Optuna:XGB      0.43 0.97 0.03 0.57
XGBClassifier   0.43 0.97 0.03 0.57
Perceptron      0.24 0.98 0.02 0.76

'''

# Cross Val -------------------------------------------------------------------
scores = cross_val_score(
    fitted_models['Optuna:XGB'],
    X_train_t_df,
    y_train,
    cv=5,
    scoring="accuracy"
 )

scores.mean()

'''
Hitting an average performance score of 92%.
'''
# -----------------------------------------------------------------------------

# Feature Importance ----------------------------------------------------------
# Get feature importance
importance = fitted_models['Optuna:XGB'].feature_importances_

# Plot feature importance
plt.figure(figsize=(16, 24))

# Create a horizontal bar plot
plt.barh([x for x in range(len(importance))], importance, color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature Index")
plt.title("Feature Importance from Optuna:XGB")
plt.yticks(range(len(importance)), labels=X_train_t_df.columns, fontsize=10)
plt.show()
# -----------------------------------------------------------------------------