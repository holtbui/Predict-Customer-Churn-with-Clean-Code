'''
Module contain function of churn customer analysis

Author : Holt Bui

Date : 2nd August 2023
'''

# import libraries
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
os.environ['QT_QPA_PLATFORM']='offscreen'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    dataframe = pd.read_csv(pth)
    return dataframe


def perform_eda(dataframe):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    #create customer age histrogram plot
    plt.figure(figsize=(20,10))
    dataframe['Customer_Age'].hist()
    plt.savefig("./images/eda/customer_age.png")

    #create maritial status plot
    plt.figure(figsize=(20,10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("./images/eda/maritial_status.png")

    #create plot to show distributions of 'Total_Trans_Ct'
    #and add a smooth curve obtained using a kernel density estimate
    plt.figure(figsize=(20,10))
    sns.histplot(dataframe['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig("./images/eda/toal_trans_ct.png")

    plt.figure(figsize=(20,10))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig("./images/eda/heat_map.png")


def encoder_helper(dataframe, category_list):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
            be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    X = pd.DataFrame()

    #adds Churn column to dataframe based on Atttrition_Flag column
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    #Maps the means of each groups in the catetory list and renames the columns to reflect churn
    for category in category_list:
        groups = dataframe.groupby(category).mean()['Churn']
        dataframe[category]=dataframe[category].map(groups)
        dataframe=dataframe.rename(columns={category:category+'_Churn'})

    #define list of columns to keep
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
         'Total_Relationship_Count', 'Months_Inactive_12_mon',
         'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
         'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
         'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
         'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
         'Income_Category_Churn', 'Card_Category_Churn','Churn']

    X[keep_cols] = dataframe[keep_cols]
    return X

def perform_feature_engineering(dataframe):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
                        used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Assign Churn column as y and drops Chrun from dataframe and assign to X
    y = dataframe['Churn']
    X = dataframe.drop('Churn', axis = 1)

    # split the X and y data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # create and save random forest classification report
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig("./images/results/rf_report.png")

    # create and save Logistic Regression classification report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/lr_report.png")

def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth+"/feature_importance.png")

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Instantiate rondom forest and logistic regresssion
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    #define grid search parameters for random forest
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    #fit random forest model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    #fit logistic regresssion model
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # create and save random forest classification report
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig("./images/results/rf__train_report.png")

    # create and save Logistic Regression classification report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/lr_train_report.png")

    #save models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
    

if __name__ == "__main__":
    dataframe = import_data("./data/bank_data.csv")
    perform_eda(dataframe)
    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]
    dataframe = encoder_helper(dataframe, cat_columns)
    X_train, X_test, y_train, y_test = perform_feature_engineering(dataframe)
    
    y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = train_models(X_train, X_test, y_train, y_test)
    
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
