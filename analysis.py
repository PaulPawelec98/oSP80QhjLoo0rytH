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

# dimension reduction
from sklearn.decomposition import TruncatedSVD

# optimization
from sklearn.model_selection import GridSearchCV

# Evaluation Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

# Ensemble
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

# Hyper Parameter Tuning
from hyperopt import fmin, tpe, hp, STATUS_OK
from hyperopt.pyll.base import scope
from hyperopt import space_eval

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
from HyperOptObj import HyperOptObj

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

# Stacked Bar Plot ------------------------------------------------------------
# Define the range of ages
age_range = range(min(df['age']), max(df['age']) + 1)

# Create DataFrames for 'yes' and 'no'
ages_yes = pd.DataFrame({
    'age': age_range,
    'y': 'yes',
    'value': 0
})

ages_no = pd.DataFrame({
    'age': age_range,
    'y': 'no',
    'value': 0
})

# Concatenate the two DataFrames
ages = pd.concat([ages_yes, ages_no], ignore_index=True)

age_group = df.groupby(['age', 'y'])['age'].count()
age_group.name = 'value'
age_group = age_group.reset_index()

df_complete = pd.merge(
    ages, age_group, on=['age', 'y'], how='outer', suffixes=('', '_original')
    )

df_complete.fillna(0, inplace=True)

df_complete['value'] = df_complete['value_original'].combine_first(
    df_complete['value']
    )

df_complete = df_complete[['age', 'y', 'value']]


age_group_no = [
    val
    for val, y
    in zip(df_complete['value'], df_complete['y'])
    if y == 'no'
    ]

age_group_yes = [
    val
    for val, y
    in zip(df_complete['value'], df_complete['y'])
    if y != 'no'
    ]

plt.bar(
        list(df_complete['age'].unique()),
        age_group_no,
        label='No',
        color='blue'
        )

plt.bar(
        list(df_complete['age'].unique()),
        age_group_yes,
        bottom=age_group_no,
        label='Yes',
        color='Orange'
        )

# Add the legend
plt.legend(title="Legend", loc="upper right")

# Titiles and Show
plt.title('Stacked Bar Chart: AGES')
plt.xlabel('Age')
plt.ylabel('Value')
plt.show()

'''
Stacked bar chart shows that most people with term deposits are around 30-40
years old.
'''
# -----------------------------------------------------------------------------

# Violin for Ages -------------------------------------------------------------
# age_pivot = age_group.pivot_table(
#     index='age', columns='y', values='value', fill_value=0
#     )

# # total_yes_no = df['y'].value_counts()

# # age_pivot['no'] = age_pivot['no']/total_yes_no['no']
# # age_pivot['yes'] = age_pivot['yes']/total_yes_no['yes']

# # Create the violin plot
# plt.figure(figsize=(10, 6))
# sns.violinplot(x="y", y="age", data=df, inner="quartile", palette="muted")

# plt.title('Violin for AGE')
# plt.show()
# '''
# Easier to see here where most term deposits are.
# However, it dosen't seem like
# a particular age matters between groups.
# '''
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
- plot confusion matrix
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

I think it's unlikely for people to convice individuals to change their risk
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
reg_results[reg6] = smf.logit(reg6, data=df).fit()
# -----------------------------------------------------------------------------

summary_col(reg_results[reg2], stars=True, float_format='%0.4f')

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

# %% [6A] Setup Data for ML
'''
hot-encoding
standardization
'''

# Setup X and y
X = df[df.columns[0:-1]]
y = df['y']
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
plt.yticks(range(len(importance)), labels=X_train_t_df.columns, fontsize=10)  # Adjust the fontsize as needed
plt.show()
# %% [6B] Setup Dimension Reduction
'''
dimension reduction
    - TruncatedSVD (Good for sparse data, for NLP?)

Some variables seem a little bit odd to use to train this model, specifically
'day' and 'month'.
    - maybe I can transform day instead? Have it be a percentage towards
    start of end of month.

Also, data is very sparse now due to encoding. So we should try and reduce.
'''

# Truncated SVD ---------------------------------------------------------------
svd = TruncatedSVD(
    n_components=min(X_train_t_df.shape)-1,
    random_state=seed
    )

svd.fit(X_train_t_df)

explained_variance = svd.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o'
    )

plt.title('Cumulative Explained Variance for SVD')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')  # Threshold line at 95%
plt.grid()

# Set x-ticks to show all components
plt.xticks(range(1, len(cumulative_variance) + 1), fontsize=8)

# Offset every other x-tick up and down
for i, tick in enumerate(plt.gca().get_xticklabels()):
    if i % 2 == 0:  # Adjust only even-indexed ticks
        tick.set_y(tick.get_position()[1] - 0.01)
    else:
        tick.set_y(tick.get_position()[1] + 0.01)

plt.show()

'''
22 components is best for this dataset based on the graph
However, there is a little bit of an elbow around 8 components, high variance
followed by low. This could be a decent spot as well.
'''
# -----------------------------------------------------------------------------

# Apply Reduction -------------------------------------------------------------
svd = TruncatedSVD(
    n_components=22,
    random_state=seed
    )

svd.fit(X_train_t_df)

X_train_t_df_dr = svd.fit_transform(X_train_t_df)
X_test_t_df_dr = svd.fit_transform(X_test_t_df)
# -----------------------------------------------------------------------------

# %% [7A] Model Selection - Normal

# drop duration
X_train_t_df = X_train_t_df.drop(columns=['pipeline-1__duration'])
X_test_t_df = X_test_t_df.drop(columns=['pipeline-1__duration'])


# LazyPredict
clf = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    custom_metric=None,
    predictions=True,
    random_state=seed
    )

lazymodels, predictions = clf.fit(X_train_t_df, X_test_t_df, y_train, y_test)
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
Seems like the data is fairly non-linear, most linear models aren't doing too
well.

These models seemed to have done the best in terms of correctly identifying
those with term deposits.

- NearestCentroid
- PassiveAggressiveClassifier
- QuadraticDiscriminantAnalysis

NearestCentriod was able to correctly identify 1117 as having a term deposits,
which is fairly close to the total 1448 in the test set, around 77%.


Same Analysis, but with duration dropped....

                               Accuracy  Balanced Accuracy  ROC AUC  F1 Score  Time Taken
Model
NearestCentroid                    0.67               0.63     0.63      0.75        0.23
GaussianNB                         0.90               0.59     0.59      0.90        0.15
ExtraTreeClassifier                0.88               0.55     0.55      0.88        0.15
DecisionTreeClassifier             0.86               0.55     0.55      0.87        0.35
LabelPropagation                   0.89               0.55     0.55      0.88       79.55
LabelSpreading                     0.89               0.55     0.55      0.88       95.28
Perceptron                         0.85               0.54     0.54      0.86        0.14
BernoulliNB                        0.93               0.54     0.54      0.90        0.15
ExtraTreesClassifier               0.92               0.54     0.54      0.90        4.47
QuadraticDiscriminantAnalysis      0.45               0.54     0.54      0.56        0.19
KNeighborsClassifier               0.93               0.54     0.54      0.90        1.96
XGBClassifier                      0.93               0.54     0.54      0.90        1.23
BaggingClassifier                  0.92               0.54     0.54      0.90        2.05
PassiveAggressiveClassifier        0.86               0.53     0.53      0.87        0.14
LinearDiscriminantAnalysis         0.93               0.53     0.53      0.90        0.32
CalibratedClassifierCV             0.93               0.53     0.53      0.90       23.07
RandomForestClassifier             0.93               0.52     0.52      0.90        4.53
LGBMClassifier                     0.93               0.52     0.52      0.90        0.68
SGDClassifier                      0.93               0.52     0.52      0.90        0.30
AdaBoostClassifier                 0.93               0.52     0.52      0.90        1.92
SVC                                0.93               0.52     0.52      0.90       55.05
LogisticRegression                 0.93               0.51     0.51      0.90        0.20
RidgeClassifier                    0.93               0.51     0.51      0.90        0.15
RidgeClassifierCV                  0.93               0.51     0.51      0.90        0.23
LinearSVC                          0.93               0.51     0.51      0.90        6.46
DummyClassifier                    0.93               0.50     0.50      0.89        0.11
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
I've ran a couple times and it seems like PassiveAggressiveClassifier might be
 a little bit inconsistent, models like DecisionTreeClassifier,
 LinearDiscriminantAnalysis, or XGBClassifier might also be good models to
 use.

                                tp   tn   fp   fn
NearestCentroid               0.75 0.89 0.11 0.25
QuadraticDiscriminantAnalysis 0.54 0.69 0.31 0.46
PassiveAggressiveClassifier   0.43 0.96 0.04 0.57
LinearDiscriminantAnalysis    0.41 0.97 0.03 0.59
Perceptron                    0.40 0.92 0.08 0.60
GaussianNB                    0.39 0.94 0.06 0.61
DecisionTreeClassifier        0.39 0.95 0.05 0.61
XGBClassifier                 0.34 0.98 0.02 0.66
AdaBoostClassifier            0.33 0.98 0.02 0.67
LGBMClassifier                0.32 0.98 0.02 0.68
ExtraTreeClassifier           0.30 0.95 0.05 0.70
BaggingClassifier             0.29 0.98 0.02 0.71
LogisticRegression            0.27 0.99 0.01 0.73
CalibratedClassifierCV        0.26 0.99 0.01 0.74
LabelPropagation              0.25 0.96 0.04 0.75
LabelSpreading                0.25 0.96 0.04 0.75
SGDClassifier                 0.22 0.99 0.01 0.78
LinearSVC                     0.20 0.99 0.01 0.80
RandomForestClassifier        0.19 0.99 0.01 0.81
ExtraTreesClassifier          0.17 0.99 0.01 0.83
BernoulliNB                   0.16 0.98 0.02 0.84
RidgeClassifier               0.14 0.99 0.01 0.86
RidgeClassifierCV             0.14 0.99 0.01 0.86
SVC                           0.14 0.99 0.01 0.86
KNeighborsClassifier          0.12 0.99 0.01 0.88
DummyClassifier               0.00 1.00 0.00 1.00

Otherwise, NearestCentroid and QuadraticDiscriminantAnalysis, both seem like
good models.

With duration dropped...

                                tp   tn   fp   fn
QuadraticDiscriminantAnalysis 0.64 0.44 0.56 0.36
NearestCentroid               0.59 0.68 0.32 0.41
GaussianNB                    0.23 0.95 0.05 0.77
Perceptron                    0.19 0.90 0.10 0.81
DecisionTreeClassifier        0.18 0.92 0.08 0.82
ExtraTreeClassifier           0.17 0.93 0.07 0.83
LabelPropagation              0.15 0.94 0.06 0.85
LabelSpreading                0.15 0.95 0.05 0.85
PassiveAggressiveClassifier   0.14 0.92 0.08 0.86
ExtraTreesClassifier          0.10 0.98 0.02 0.90
BernoulliNB                   0.09 0.99 0.01 0.91
KNeighborsClassifier          0.08 0.99 0.01 0.92
BaggingClassifier             0.08 0.99 0.01 0.92
XGBClassifier                 0.08 0.99 0.01 0.92
LinearDiscriminantAnalysis    0.06 1.00 0.00 0.94
CalibratedClassifierCV        0.06 1.00 0.00 0.94
RandomForestClassifier        0.05 1.00 0.00 0.95
LGBMClassifier                0.05 1.00 0.00 0.95
SGDClassifier                 0.04 1.00 0.00 0.96
AdaBoostClassifier            0.04 1.00 0.00 0.96
SVC                           0.03 1.00 0.00 0.97
LogisticRegression            0.03 1.00 0.00 0.97
RidgeClassifier               0.03 1.00 0.00 0.97
RidgeClassifierCV             0.03 1.00 0.00 0.97
LinearSVC                     0.02 1.00 0.00 0.98
DummyClassifier               0.00 1.00 0.00 1.00

'''
# -----------------------------------------------------------------------------

# %% [7B] Model Selection - Dimension Reduction

# # LazyPredict
# clf = LazyClassifier(
#     verbose=0,
#     ignore_warnings=True,
#     custom_metric=None,
#     predictions=True,
#     random_state=seed
#     )

# lazymodels, predictions = clf.fit(
#     X_train_t_df_dr,
#     X_test_t_df_dr,
#     y_train,
#     y_test
#     )

# lazymodels = pd.DataFrame(lazymodels)

# # Figure for cms
# fig_cm, axes = plt.subplots(9, 3, figsize=(24, 48))

# cm_results = {}

# # Get predictions for each model
# for i, ax in enumerate(axes.flat):
#     try:
#         # get model and pred data
#         model_name = predictions.keys()[i]
#         y_pred = predictions[model_name]

#         # Generate Confusion Matrix
#         cm = confusion_matrix(y_test, y_pred)
#         cm_results[model_name] = cm

#         # Plot Confusion Matrix
#         sns.heatmap(
#             cm,
#             annot=True,
#             fmt='d',
#             cmap='Blues',
#             xticklabels=['Pred: 0', 'Pred: 1'],
#             yticklabels=['True: 0', 'True: 1'],
#             ax=ax,
#             cbar=False  # Remove the colorbar
#             )

#         ax.set_title(f'Confusion Matrix for {model_name}')
#         ax.set_xlabel('Predicted Label')
#         ax.set_ylabel('True Label')

#     except Exception:
#         ax.axis('off')

# fig_cm.suptitle("Confusion Matrix Heatmap: LazyPredict", fontsize=16, y=1.02)
# plt.tight_layout()
# plt.show()

# '''
# Similar results to before. Same models are performing the best.
# '''
# # -----------------------------------------------------------------------------

# Closer Look at CM Results ---------------------------------------------------
# cm_ratios_df = pd.DataFrame(columns=['tp', 'tn', 'fp', 'fn'])

# for key, value in cm_results.items():

#     tn, fp, fn, tp = value.flatten()

#     cm_ratios_df.loc[key] = {
#             'tp': tp / (tp + fn) if (tp + fn) > 0 else 0,
#             'tn': tn / (tn + fp) if (tn + fp) > 0 else 0,
#             'fp': fp / (fp + tn) if (fp + tn) > 0 else 0,
#             'fn': fn / (fn + tp) if (fn + tp) > 0 else 0
#         }

# cm_ratios_df.sort_values(by='tp', ascending=False)

'''

At 22 components....

                                tp   tn   fp   fn
NearestCentroid               0.69 0.90 0.10 0.31
QuadraticDiscriminantAnalysis 0.46 0.97 0.03 0.54
PassiveAggressiveClassifier   0.40 0.94 0.06 0.60
LinearDiscriminantAnalysis    0.39 0.97 0.03 0.61
DecisionTreeClassifier        0.35 0.94 0.06 0.65
GaussianNB                    0.34 0.97 0.03 0.66
LabelPropagation              0.34 0.94 0.06 0.66
LabelSpreading                0.34 0.94 0.06 0.66
XGBClassifier                 0.30 0.98 0.02 0.70
ExtraTreeClassifier           0.30 0.90 0.10 0.70
LGBMClassifier                0.29 0.98 0.02 0.71
AdaBoostClassifier            0.29 0.98 0.02 0.71
LogisticRegression            0.23 0.99 0.01 0.77
CalibratedClassifierCV        0.23 0.99 0.01 0.77
Perceptron                    0.23 0.98 0.02 0.77
KNeighborsClassifier          0.23 0.98 0.02 0.77
BaggingClassifier             0.21 0.99 0.01 0.79
RandomForestClassifier        0.18 0.99 0.01 0.82
SVC                           0.16 0.99 0.01 0.84
LinearSVC                     0.15 0.99 0.01 0.85
RidgeClassifier               0.11 0.99 0.01 0.89
RidgeClassifierCV             0.11 0.99 0.01 0.89
ExtraTreesClassifier          0.08 1.00 0.00 0.92
SGDClassifier                 0.07 1.00 0.00 0.93
BernoulliNB                   0.06 0.99 0.01 0.94
DummyClassifier               0.00 1.00 0.00 1.00

Pretty similar results from before, seems like nearestcentroid is doing really
good with this kind of data.

At 8 components...

                                tp   tn   fp   fn
NearestCentroid               0.68 0.90 0.10 0.32
QuadraticDiscriminantAnalysis 0.39 0.97 0.03 0.61
LinearDiscriminantAnalysis    0.39 0.97 0.03 0.61
DecisionTreeClassifier        0.36 0.95 0.05 0.64
GaussianNB                    0.33 0.97 0.03 0.67
LabelPropagation              0.32 0.96 0.04 0.68
LabelSpreading                0.31 0.96 0.04 0.69
LGBMClassifier                0.30 0.98 0.02 0.70
ExtraTreeClassifier           0.30 0.95 0.05 0.70
AdaBoostClassifier            0.30 0.98 0.02 0.70
XGBClassifier                 0.28 0.98 0.02 0.72
KNeighborsClassifier          0.23 0.98 0.02 0.77
BaggingClassifier             0.23 0.98 0.02 0.77
RandomForestClassifier        0.23 0.98 0.02 0.77
LogisticRegression            0.22 0.99 0.01 0.78
CalibratedClassifierCV        0.22 0.99 0.01 0.78
ExtraTreesClassifier          0.19 0.99 0.01 0.81
SVC                           0.17 0.99 0.01 0.83
Perceptron                    0.17 0.98 0.02 0.83
LinearSVC                     0.14 0.99 0.01 0.86
RidgeClassifier               0.12 0.99 0.01 0.88
RidgeClassifierCV             0.12 0.99 0.01 0.88
SGDClassifier                 0.01 1.00 0.00 0.99
PassiveAggressiveClassifier   0.01 1.00 0.00 0.99
DummyClassifier               0.00 1.00 0.00 1.00
BernoulliNB                   0.00 1.00 0.00 1.00

Pretty much the same results, except for PassiveAggressiveClassifier falling
off entirely.
'''
# -----------------------------------------------------------------------------

# %% [8] Initial Model Performance and Tuning

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
#     scoring='accuracy',
#     n_jobs=-1
#     )

# model.fit(X_train_t_df, y_train)

# fitted_models['NearestCentroid'] = model
# -----------------------------------------------------------------------------

# PassiveAggressiveClassifier -------------------------------------------------
# param_grid = {
#     'C': [0.01, 0.1, 1, 10],  # Regularization strength
#     'max_iter': [500, 1000, 2000],  # Iteration limits
#     'tol': [1e-3, 1e-4],  # Tolerance for stopping
#     'loss': ['hinge', 'squared_hinge']  # Hinge loss options
# }

# model = GridSearchCV(
#     PassiveAggressiveClassifier(),
#     param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
#     )

# model = model.fit(X_train_t_df, y_train)
# fitted_models['PassiveAggressiveClassifier'] = model
# -----------------------------------------------------------------------------

# QuadraticDiscriminantAnalysis -----------------------------------------------
# param_grid = {
#     'reg_param': [0.0, 0.01, 0.1, 0.5, 1.0],  # Regularization strength
#     'tol': [1e-4, 1e-3, 1e-2],  # Convergence tolerance
#     'store_covariance': [True, False]  # Whether to store covariance
# }

# model = GridSearchCV(
#     QuadraticDiscriminantAnalysis(),
#     param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
#     )

# model = model.fit(X_train_t_df, y_train)
# fitted_models['QuadraticDiscriminantAnalysis'] = model
# -----------------------------------------------------------------------------

# Models with Best Params -----------------------------------------------------
'''
All selected models setup via best parameters from prior GridsearchCV runs...
'''

selected_models = [
    NearestCentroid,
    PassiveAggressiveClassifier,
    QuadraticDiscriminantAnalysis
    ]

params = {
    'NearestCentroid': {
        'metric': 'euclidean', 'shrink_threshold': 0.2
        },
    'PassiveAggressiveClassifier': {
        'C': 0.01, 'loss': 'squared_hinge', 'max_iter': 500, 'tol': 0.001
        },
    'QuadraticDiscriminantAnalysis': {
        'reg_param': 1.0, 'store_covariance': True, 'tol': 0.0001
        }
    }

for model, name in zip(selected_models, params):
    result = model(**params[name])
    result = result.fit(X_train_t_df, y_train)
    fitted_models[name] = result
# -----------------------------------------------------------------------------

# Copy Models for Ensemble ----------------------------------------------------
voting_models = fitted_models.copy()
voting_models = list(voting_models.items())
# -----------------------------------------------------------------------------

# Voting ----------------------------------------------------------------------
model = VotingClassifier(
    estimators=voting_models,
    voting='hard',
    )

model = model.fit(X_train_t_df, y_train)
fitted_models['Voting'] = model
# -----------------------------------------------------------------------------

# Stacking --------------------------------------------------------------------
meta_model = LogisticRegression()
model = StackingClassifier(
    estimators=voting_models, final_estimator=meta_model
    )

model = model.fit(X_train_t_df, y_train)
fitted_models['Stacking'] = model
# -----------------------------------------------------------------------------

'''
Maybe I could add in more models just for voting? Our models are fairly simple,
maybe we could tune a more complicated model better?
'''

# %% [9] Model Results

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

# cv_results = {
#     x: fitted_models[x].cv_results_
#     for x in fitted_models if x not in ['Voting', 'Stacking']
# }

# best_parameters = {
#     x: fitted_models[x].best_params_
#     for x in fitted_models if x not in ['Voting', 'Stacking']
#     }

'''
It seems like I'm able to hit that average performance score of 81%?

> cv_results['NearestCentroid']['mean_test_score'].mean()
> 0.8200312500000001

Otherwise, most models are fairly accurate with accuracy scores as high as 93%

> accuracy
> {'NearestCentroid': 0.8872142857142857,
  'PassiveAggressiveClassifier': 0.9315714285714286,
  'QuadraticDiscriminantAnalysis': 0.9344285714285714,
  'Voting': 0.9347142857142857,
  'Stacking': 0.9313928571428571}
'''
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
Seems like NearestCentroid is the best model?
'''

# %% [10] HyperTune model for best true positive rates

# Optimize --------------------------------------------------------------------
'''
Optimize NearestCentroid for better true positive results.
'''


def tp_score(self, model):
    y_pred = model.predict(self.X_test)
    y_pred = (y_pred >= 0.5).astype(int)
    result = confusion_matrix(self.y_test, y_pred)

    tn, fp, fn, tp = result.flatten()

    return tp / (tp + fn) if (tp + fn) > 0 else 0


def tp_fp_score(self, model):
    y_pred = model.predict(self.X_test)
    y_pred = (y_pred >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).flatten()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Minimizes FP
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0      # Maximizes TP

    score = ((2 * precision * recall) / (precision + recall)
             if (precision + recall) > 0 else 0)

    return score


def default_score(self, model):

    y_pred = model.predict(self.X_test)
    y_pred = (y_pred >= 0.5).astype(int)

    # score = matthews_corrcoef(self.y_test, y_pred)
    score = model.score(self.X_test, self.y_test)

    return score


space = {
    'metric': hp.choice('metric', ['euclidean', 'manhattan']),
    'shrink_threshold': hp.quniform('shrink_threshold', 0.001, 0.5, 0.001)
}

HypObj = HyperOptObj(X_train_t_df,
                     X_test_t_df,
                     y_train,
                     y_test,
                     NearestCentroid,
                     default_score
                     )

opt_results = HypObj.run_optimization(space)
# -----------------------------------------------------------------------------

model = NearestCentroid(**opt_results)
model = model.fit(X_train_t_df, y_train)

y_pred = model.predict(X_test_t_df)
y_pred = (y_pred >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

'''
Optimizing using TP score gave us...

array([[16708,  1844],
       [  454,   994]], dtype=int64)

Which is decent, but a pretty large increase in false positives as well.


Optimizing for TP/FP scores....

array([[23462,  2511],
       [  638,  1389]], dtype=int64)

If I tune for better shrinkage, I don't get much of an improvement

If I try for other scores like matthews or default I don't see much
improvments
'''
