'''
Module contains tests and logging of churn_library functions

Author : Holt Bui

Date : 2nd August 2023
'''

import os
import logging
import pytest
import churn_library as cl


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def df_plugin():
    """
    Create pystest Namespace
    """
    return None

def pytest_configure():
    """
    Create Dataframe object in Namespace
    """
    pytest.df = df_plugin()

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = cl.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    pytest.dataframe = dataframe

def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    dataframe = pytest.dataframe
    cl.perform_eda(dataframe)
    graphs = ["customer_age", "maritial_status", "toal_trans_ct", "heat_map"]
    for graph in graphs:
        try:
            assert os.path.isfile("./images/eda/"+graph+".png")
            logging.info("%s exists", graph)
        except AssertionError:
            logging.error("%s Does not exist", graph)

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    dataframe = pytest.dataframe
    cat_columns = ['Gender',
                   'Education_Level',
                   'Marital_Status',
                   'Income_Category',
                   'Card_Category']

    dataframe = cl.encoder_helper(dataframe, cat_columns)
    try:
        assert dataframe['Churn'].dtype == 'int64'
        logging.info("Churn mapped from Attribute")
    except KeyError:
        logging.error("Churn column does not exit")
    except AssertionError:
        logging.error("Churn column datatype is not int64")

    for column in cat_columns:
        try:
            assert column not in dataframe.columns
            logging.info("%s has been dropped", column)
        except AssertionError:
            logging.error("%s has not been dropped", column)

    for column in cat_columns:
        try:
            encoded_column = column+"_Churn"
            assert dataframe[encoded_column].dtype == 'float'
            logging.info("%s encoded to float as %s", column, encoded_column)
        except AssertionError:
            logging.info("%s has not been encoded to float as %s", column, encoded_column)

    pytest.dataframe = dataframe

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    dataframe = pytest.dataframe
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(dataframe)

    df_names= {
        0: 'X_train',
        1: 'X_test',
        2: 'y_train',
        3: 'y_test'
    }
    for i, df in enumerate([X_train, X_test, y_train, y_test]):
        try:
            for j, dim in enumerate(dataframe.shape):
                assert dataframe.shape[j] > 0
                if j==0:
                    logging.info("%s engineered with %d rows", df_names[i], dataframe.shape[j])
                    #logging.info(df_names[i]+" engineered with "+str(dataframe.shape[j])+" rows")
                else:
                    logging.info("%s engineered with %d columns",
                                 df_names[i], dataframe.shape[j])
                    #logging.info(df_names[i]+" 
                    #engineered with "+str(dataframe.shape[j])+" columns")
        except  AssertionError:
            logging.error("%s was not engineered correctly", df_names[i])

    pytest.X_train = X_train
    pytest.X_test = X_test
    pytest.y_train = y_train
    pytest.y_test = y_test

def test_train_models(train_models):
    '''
    test train_models
    '''
    X_train  = pytest.X_train
    X_test = pytest.X_test
    y_train = pytest.y_train
    y_test = pytest.y_test
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    cl.train_models(X_train, X_test, y_train, y_test)
    try:
        assert os.path.isfile('./models/rfc_model.pkl')
        logging.info("Random forest model has been trained and saved")
    except AssertionError:
        logging.error("ERROR: random forest model does not exist on the file system")

    try:
        assert os.path.isfile('./models/logistic_model.pkl')
        logging.info("Logistic regression model has been trained and saved")
    except AssertionError:
        logging.error("ERROR: Logistic regression does not exist on the file system")

if __name__ == "__main__":
    test_import("./data/bank_data.csv")
    test_eda(test_import("./data/bank_data.csv"))
    test_encoder_helper(test_import("./data/bank_data.csv"))
    test_perform_feature_engineering(
         test_encoder_helper(
         test_import("./data/bank_data.csv")))
#    test_train_models(
#         test_perform_feature_engineering(
#         test_encoder_helper(
#         test_import("./data/bank_data.csv"))))
