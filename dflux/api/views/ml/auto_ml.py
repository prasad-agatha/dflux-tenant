import sqlite3
import io
import os
import pickle
import random
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import requests
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from sklearn.decomposition import PCA
from scipy.stats import zscore

warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import (
    LeavePOut,
    LeaveOneOut,
    train_test_split,
    ShuffleSplit,
    KFold,
    StratifiedKFold,
    GridSearchCV as GCV,
)
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from mlxtend.preprocessing import minmax_scaling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
import math
import xgboost as xg
from sklearn.svm import SVR
from typing import Tuple

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pmdarima.arima import auto_arima
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import uuid

# 1. LOADING

# 1.1
def read_data_from_db(table_query, conn):
    """
    Read data from database and pull to console as DataFrame
    Input Parameters:
                    a. sqlite3_query
                    b. cur_execute_query
                    c. table_query
    Returns: DataFrame
    """
    # Create a SQL connection to our SQLite database
    # con = sqlite3.connect(sqlite3_query)
    # cur = con.cursor()
    # # Be sure to close the connection
    # cur.execute(cur_execute_query)
    # print(cur.fetchall())
    df = pd.read_sql_query(table_query, conn)
    # cur.close()
    # con.close()
    return df


# 1.2
def read_data(csv_file_path):
    """
    Read CSV file
    Input Parameters: File Path (.csv)
    Returns: DataFrame
    """
    df = pd.read_csv(csv_file_path)
    return df


# 2. PREPROCESSING
# 2.1
def extract_numeric_and_categorical_column_names(
    data: pd.DataFrame,
) -> Tuple[list, list]:
    numeric_columns = data.select_dtypes(
        ["float64", "int64", "int16", "float16", "float32", "int32", "uint8"]
    ).columns
    categorical_columns = [
        column_name
        for column_name in data.columns
        if column_name not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def data_type_correction(df):
    """
    Performs auto correction of data type of each column and if date time column present in the data, this function will convert those columns to datetime format and extract
    1) day
    2) month
    3) year
    4) hour
    5) minute
    6) seconds
    7) name of the day
    Input Parameters: DataFrame
    Returns: DataFrame
    """
    _, categorical_columns = extract_numeric_and_categorical_column_names(df)
    if len(categorical_columns) != 0:
        date_time_columns = []
        # filtering date time column.
        from pandas.errors import ParserError

        for column in categorical_columns:
            try:
                df[column] = pd.to_datetime(df[column])
                date_time_columns.append(column)
            except (ParserError, ValueError):
                pass
        # dropping datetime column names from the categorical column list and adding few new columns in the dataset.
        if len(date_time_columns) != 0:
            for column in date_time_columns:
                categorical_columns.remove(column)
                # adding few new columns based on the datetime type columns
                data_column_functions = (
                    lambda column_data: column_data.dt.day,
                    lambda column_data: column_data.dt.month,
                    lambda column_data: column_data.dt.year,
                    lambda column_data: column_data.dt.hour,
                    lambda column_data: column_data.dt.minute,
                    lambda column_data: column_data.dt.second,
                    lambda column_data: column_data.dt.day_name(),
                )
                order = {
                    0: "day",
                    1: "month",
                    2: "year",
                    3: "hour",
                    4: "minute",
                    5: "second",
                    6: "day_names",
                }
                for index, fun in enumerate(data_column_functions):
                    try:
                        df[column + "_" + order[index]] = fun(df[column]).astype(
                            "category"
                        )
                    except (ParserError, ValueError):
                        pass
            # deleting older datetime column
            df.drop(labels=date_time_columns, axis=1, inplace=True)
        # converting the rest of the categorical columns to string type.
        df[categorical_columns] = df[categorical_columns].astype("category")
        if "Unnamed: 0" in list(df.columns):
            df.drop("Unnamed: 0", axis=1, inplace=True)
    return df


# 2.2
def auto_remove_unwanted_columns(df):
    """
    Performs auto removal of columns which are not useful for model training
    Input Parameters: DataFrame
    Returns: DataFrame
    """
    _, categorical_columns = extract_numeric_and_categorical_column_names(df)
    for column_name in categorical_columns:
        if int(df[column_name].nunique()) == 1 or int(df[column_name].nunique()) == len(
            df[column_name]
        ):
            df = df.drop(column_name, axis=1)
    if "Unnamed: 0" in list(df.columns):
        df.drop("Unnamed: 0", axis=1, inplace=True)
    return df


# 2.3
def auto_imputer(df):
    """
    Imputation of empty enteries by central tendency: Mean and Mode
    Input Parameters: DataFrame
    Returns: DataFrame
    """
    numeric_columns, categorical_columns = extract_numeric_and_categorical_column_names(
        df
    )
    if len(categorical_columns) != 0 and len(numeric_columns) != 0:
        # imputing categorical columns
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        categorical_data = pd.DataFrame(
            categorical_imputer.fit_transform(np.array(df[categorical_columns])),
            columns=categorical_columns,
        )
        # imputing numeric columns
        numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        numeric_data = pd.DataFrame(
            numeric_imputer.fit_transform(np.array(df[numeric_columns])),
            columns=numeric_columns,
        )
        # combining these columns into one dataframe
        df = pd.concat([numeric_data, categorical_data], axis=1)
    elif len(categorical_columns) != 0:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        df = pd.DataFrame(
            categorical_imputer.fit_transform(np.array(df[categorical_columns])),
            columns=categorical_columns,
        )
    elif len(numeric_columns) != 0:
        numeric_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        df = pd.DataFrame(
            numeric_imputer.fit_transform(np.array(df[numeric_columns])),
            columns=numeric_columns,
        )
    else:
        pass
    if "Unnamed: 0" in list(df.columns):
        df.drop("Unnamed: 0", axis=1, inplace=True)
    df = data_type_correction(df)
    return df


# 2.4
def remove_correlated_columns(df, target_column_name=None, threshold=None):
    """
    Auto Removal of hight co-related columns based on user threshold.
    Input Parameters:
                    a. DataFrame
                    b. Threshold: numeric attribute
    Returns: DataFrame
    """
    if target_column_name is not None:
        target = df[target_column_name]
        df = df.drop(target_column_name, axis=1)
    # Create correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop features
    df.drop(to_drop, axis=1, inplace=True)
    if target_column_name is not None:
        df[target_column_name] = target
    return df


"""
Detecting Outliers
"""

"""
statistic methods used for outliers detection.
"""
from typing import Callable, Any, List, Dict, Union
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import numpy as np
import scipy.stats as stats


# outlier detections by zsocres
def detecting_outliers_by_Zscore(
    data: pd.DataFrame, target_variable: str = None, threshold_zscore: float = 3.0
) -> pd.DataFrame:
    """
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    description : This Function will take data and threshold zscore as input and it will return list of dictionary, where key would be column name and value would be list of indexes of the outlier in that column
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) data :-   This should be pandas dataframe which should be cleaned and contain numeric values.
    2) threshold_zscore :- any zscore which programmer would like to make it as breaking point for finding outliers in the disturbution.
    ---------------------------------------------------------------------
    returns :- List of dictionaries where key of the dictionary is column name and value are list of outlier indexes
    ---------------------------------------------------------------------
    """
    numeric_columns, _ = extract_numeric_and_categorical_column_names(data)
    if target_variable != None:
        numeric_columns.remove(target_variable)
    # extracting outliers indexs based on threshold
    get_outlier_indexes_for_a_column: Callable[
        [pd.Series], np.array
    ] = lambda column: np.array(
        [
            index
            for index in range(0, len(stats.zscore(data[column])))
            if stats.zscore(data[column])[index] > threshold_zscore
            or stats.zscore(data[column])[index] < (-1 * threshold_zscore)
        ]
    )
    indexes: list = list(map(get_outlier_indexes_for_a_column, numeric_columns))
    indexes: list = [x for l in indexes for x in l]
    data.drop(indexes, inplace=True)
    return data


# outlier detection by Inter quantile ranges
def detecting_outliers_by_IOR(
    data: pd.DataFrame, target_variable: str = None
) -> pd.DataFrame:
    """
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    description : This Function will take data as input and get extract index of the outlier values for each column by IQR filter method.
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) data :-   This should be pandas dataframe which should be cleaned and contain numeric values.
    ---------------------------------------------------------------------
    returns :- List of dictionaries where key of the dictionary is column name and value are list of outlier indexes.
    ---------------------------------------------------------------------
    """
    numeric_columns, _ = extract_numeric_and_categorical_column_names(data)
    if target_variable != None:
        numeric_columns.remove(target_variable)

    def get_bounding_values_by_IQR(datacolumn: pd.Series):
        """
        ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        description : This Function will take a column and compute the upper cutoff and lower cutoff for that column by below formula
        uppercutoff = Q3 - (1.5*IQR)
        lowercutoff = Q1 - (1.5*IQR)
        ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        arguments :
        1) data :- The input should be sequence which contain numeric values.
        ---------------------------------------------------------------------
        returns :- List of dictionaries where key of the dictionary is column name and value are list of outlier indexes.
        ---------------------------------------------------------------------
        """
        sorted(datacolumn)
        Q1, Q3 = np.percentile(datacolumn, [25, 75])
        IQR: float = Q3 - Q1
        lower_range: float = Q1 - (1.5 * IQR)
        upper_range: float = Q3 + (1.5 * IQR)
        return round(lower_range, 2), round(upper_range, 2)

    extract_indexes_of_outliers: Callable[[pd.Series], list] = lambda column: [
        index
        for index in range(len(data[column]))
        if data[column][index] < get_bounding_values_by_IQR(data[column])[0]
        or data[column][index] > get_bounding_values_by_IQR(data[column])[1]
    ]
    indexes: list = list(map(extract_indexes_of_outliers, numeric_columns))
    indexes: list = [x for l in indexes for x in l]
    data.drop(indexes, axis=0, inplace=True)
    return data


"""
ML algorithm used for outliers detection.
"""
# Local outlier factor.
def detecting_outliers_by_local_outlier_factor(
    data: pd.DataFrame, target_variable: str = None
) -> pd.DataFrame:
    """
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    description : This function detect outlier by using Unspervised model named Local Outlier factor, by taking entire data at once.
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) data :-   This should be pandas dataframe which should be cleaned and contain numeric values.
    ---------------------------------------------------------------------
    returns :- List of dictionaries where key of the dictionary is column name and value are list of outlier indexes.
    ---------------------------------------------------------------------
    """
    numeric_columns, _ = extract_numeric_and_categorical_column_names(data)
    if target_variable != None:
        numeric_columns.remove(target_variable)
    model = LocalOutlierFactor(n_neighbors=2)
    extract_indexes_of_outliers: Callable[[pd.Series], list] = lambda column: [
        index
        for index, value in enumerate(
            model.fit_predict(
                np.reshape(np.array(data[column]), (len(np.array(data[column])), 1))
            )
        )
        if value == -1
    ]
    indexes: list = list(map(extract_indexes_of_outliers, numeric_columns))
    indexes: list = [x for l in indexes for x in l]
    data.drop(indexes, axis=0, inplace=True)
    return data


# IsolationForest
def detecting_outliers_by_isolation_forest(
    data: pd.DataFrame, target_variable: str = None
) -> pd.DataFrame:
    """
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    description : This function detect outlier by using ensembled model named Isolation_forest, by taking entire data at once.
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) data :-   This should be pandas dataframe which should be cleaned and contain numeric values.
    ---------------------------------------------------------------------
    returns :- List of row indices which are outliers.
    ---------------------------------------------------------------------
    """
    numeric_columns, _ = extract_numeric_and_categorical_column_names(data)
    if target_variable != None:
        numeric_columns.remove(target_variable)
    data_numeric: pd.DataFrame = data[numeric_columns]
    model = IsolationForest(n_estimators=20, warm_start=True)
    model.fit(data_numeric.values)
    indexes: np.array = np.array(
        [
            index
            for index, value in enumerate(model.predict(data_numeric.values))
            if value == -1
        ]
    )
    if len(indexes) != 0:
        data.drop(indexes, axis=0, inplace=True)
        return data
    else:
        return data


# covariance.EllipticEnvelope
def detecting_outliers_by_elliptic_envelope(
    data: pd.DataFrame, target_variable: str = None
) -> pd.DataFrame:
    """
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    description : This function detect outlier by using EllipticEnelope model named Isolation_forest, by taking entire data at once.
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) data :-   This should be pandas dataframe which should be cleaned and contain numeric values.
    ---------------------------------------------------------------------
    returns :- returns the dataframe without outliers
    ---------------------------------------------------------------------
    """
    numeric_columns, _ = extract_numeric_and_categorical_column_names(data)
    if target_variable != None:
        numeric_columns.remove(target_variable)
    data_numeric: pd.DataFrame = data[numeric_columns]
    model = EllipticEnvelope(random_state=0).fit(data_numeric.values)
    indexes: np.array = np.array(
        [
            index
            for index, value in enumerate(model.predict(data_numeric.values))
            if value == -1
        ]
    )
    if len(indexes) == 0:
        return data
    else:
        return data.drop(indexes, axis=0, inplace=False)


def drop_duplicate_columns_rows_and_unique_value_columns(data: pd.DataFrame):
    """
    ---------------------------------------------------------------------------------------------------------------------------------
    description : This function get pandas dataFrame as an argument and drop duplicates rows and columns and unique value columns.
    ---------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) data :- This should be the pandas dataFrame which contains any type of data.
    ---------------------------------------------------------------------
    returns :- data
    ---------------------------------------------------------------------
    """
    # dropping duplicate rows
    data = data.drop_duplicates()
    # dropping duplicate columns
    data = data.T.drop_duplicates().T
    # dropping unique values
    unique_value_columns = [
        column for column in data.columns if data.shape[0] == data[column].nunique()
    ]
    data = data.drop(unique_value_columns, axis=1)
    return data


"""
categorical variable encoding techs
"""


def label_encoding(data: pd.DataFrame, categorical_column) -> pd.DataFrame:
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(data[categorical_column])
    data[categorical_column] = label_encoder.transform(data[categorical_column])
    return data, label_encoder.classes_


def one_hot_encoding(data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    for column in categorical_columns:
        dummies = pd.get_dummies(data[column], prefix=column, drop_first=False)
        data = data.drop([column], axis=1)
        data = pd.concat([data, dummies], axis=1)
    return data


def ordinal_encoder(data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    ordinal_encoder = OrdinalEncoder()
    for column in categorical_columns:
        values = data[column].values.reshape(-1, 1)
        data[column] = ordinal_encoder.fit_transform(values)
    return data


def categorical_value_encoder_transformer(
    df, encoder_transform, target_column_name=None
) -> pd.DataFrame:
    target = 0
    if target_column_name != None:
        target = df[target_column_name]
        df = df.drop(target_column_name, axis=1)
    _, categorical_columns = extract_numeric_and_categorical_column_names(df)
    if len(categorical_columns) != 0:
        df = encoder_transform(df, categorical_columns)
    if target_column_name != None:
        df[target_column_name] = target
    return df


# # 2.5
# # changes
# def categorical_value_encoder(df, categorical_column_name, encoding_type):
#     """
#     Encodes categorical attributes to numerical attributes
#     Input Parameters:
#                      a. DataFrame
#                      b. Target Variable: 'target_variable'
#                      c. Encoding Type : 'label' , 'onehot',
#     Returns: DataFrame
#     """
#     label_encoder = preprocessing.LabelEncoder()
#     target = df[categorical_column_name]
#     df = df.drop(categorical_column_name, axis=1)
#     numeric_columns = df.select_dtypes(
#         ["float64", "int64", "int16", "float16", "float32", "int32"]
#     ).columns
#     categorical_columns = [
#         column_name for column_name in df.columns if column_name not in numeric_columns
#     ]
#     for column in categorical_columns:
#         if encoding_type == "label":
#             label_encoder.fit(list(df[column]))
#             df[column] = label_encoder.transform(list(df[column]))
#         elif encoding_type == "onehot":
#             dummies = pd.get_dummies(df[column], prefix=column, drop_first=False)
#             df = pd.concat([df, dummies], axis=1)
#             df = df.drop([column], axis=1)
#         # elif encoding_type == ""
#     df[categorical_column_name] = target
#     return df

"""
Numeric transformations
"""
# 2.7
def standard_scaler(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(df)
    scaler_url = save_model_to_s3("std_sclaer", scaler)
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df, scaler_url


def robust_scaler(df: pd.DataFrame) -> pd.DataFrame:
    scaler = RobustScaler()
    for column in df.columns:
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
        scaler_url = save_model_to_s3("robust_scaler", scaler)
    return df, scaler_url


# min_max_scaling = lambda df: minmax_scaling(df, df.columns)

cube_root_transformation = lambda df: pd.DataFrame(
    np.cbrt(np.array(df)), columns=df.columns
)

square_root_transformation = lambda df: pd.DataFrame(
    np.sqrt(np.array(df)), columns=df.columns
)

log_transformation = lambda df: pd.DataFrame(np.log(np.array(df)), columns=df.columns)

square_transformation = lambda df: pd.DataFrame(
    np.square(np.array(df)), columns=df.columns
)


def numeric_transformations(
    df, transformation_function, target_column_name=None
) -> pd.DataFrame:
    target = 0
    if target_column_name != None:
        target = df[target_column_name]
        df = df.drop(target_column_name, axis=1)
    numeric_columns, _ = extract_numeric_and_categorical_column_names(df)
    if len(numeric_columns) != 0:
        transformed_data = transformation_function(df[numeric_columns])
        for column in numeric_columns:
            df[column] = transformed_data[column]
        # checking for infinity values
        if np.isinf(df).values.sum() != 0:
            from numpy import inf

            df[df == -inf] = 0
        if target_column_name != None:
            df[target_column_name] = target
        return df
    else:
        if target_column_name != None:
            df[target_column_name] = target
        return df


# 2.6
# save scalar.
def standard_scale(df, target_variable):
    """
    Normalization of numeric type columns based on standard scaler method.
    Input Parameters:
                    a. DataFrame
                    b. Target Variable: 'target_variable'
    Returns: DataFrame
    """
    numeric_columns = df.select_dtypes(
        ["float64", "int64", "int16", "float16", "float32", "int32", "uint8"]
    ).columns
    try:
        numeric_columns = numeric_columns.drop(target_variable)
    except:
        pass
    scaler = StandardScaler()
    scaler.fit(df[numeric_columns])

    scaler_url = save_model_to_s3(f"standard_scaler_{uuid.uuid4().hex}", scaler)
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    return df, scaler_url


# 2.7
def min_max_scale(df, target_variable):
    """
    Normalization of numeric type columns based on min-max scaler method.
    Input Parameters:
                    a. DataFrame
                    b. Target Variable: 'target_variable'
    Returns: DataFrame
    """
    numeric_columns = df.select_dtypes(
        ["float64", "int64", "int16", "float16", "float32", "int32", "uint8"]
    ).columns
    try:
        numeric_columns = numeric_columns.drop(target_variable)
    except:
        pass
    scaler = MinMaxScaler()
    scaler.fit(df[numeric_columns])
    scaler_url = save_model_to_s3(f"minmax_scaler_{uuid.uuid4().hex}", scaler)
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    return df, scaler_url


def robust_scale(df, target_variable):
    """
    Normalization of numeric type columns based on min-max scaler method.
    Input Parameters:
                    a. DataFrame
                    b. Target Variable: 'target_variable'
    Returns: DataFrame
    """
    numeric_columns = df.select_dtypes(
        ["float64", "int64", "int16", "float16", "float32", "int32", "uint8"]
    ).columns
    try:
        numeric_columns = numeric_columns.drop(target_variable)
    except:
        pass
    scaler = MinMaxScaler()
    scaler.fit(df[numeric_columns])
    scaler_url = save_model_to_s3(f"minmax_scaler_{uuid.uuid4().hex}", scaler)
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    return df, scaler_url


"""
Feature Generation
"""
# PCA
def fit_pca_model(data: pd.DataFrame, no_Of_features_to_generate: int) -> PCA:
    """
    ---------------------------------------------------------------------------------------------------------------------------------
    description : This function get pandas dataFrame as an argument, fit that data to PCA model and will return the PCA mode Object
    ---------------------------------------------------------------------------------------------------------------------------------
    arguments :

    1) data :- This should be the pandas dataFrame which contains no NA's and non numeric values.
    2) no_of_features_to_generate :- (int) enter the number features required after transformation of data.
    ---------------------------------------------------------------------
    returns :- PCA object
    ---------------------------------------------------------------------
    """
    pca_model: PCA = PCA(no_Of_features_to_generate, random_state=21)
    pca_model.fit(data)
    return pca_model


def load_pca_model(path: str) -> PCA:
    """
    ---------------------------------------------------------------------------------------------------------------------------------
    description : This Function will take the model path, load the model from the path and will return the model(PCA object)
    ---------------------------------------------------------------------------------------------------------------------------------
    arguments :

    1) path :- Path of the model.
    ---------------------------------------------------------------------
    returns :- PCA object
    ---------------------------------------------------------------------
    """
    loaded_model = pickle.load(open(path, "rb"))
    return loaded_model


def save_pca_model(model: PCA, path: str) -> None:
    """
    ---------------------------------------------------------------------------------------------------------------------------------
    description : This Function will take the model path and PCA mode object and save the model in the given path
    ---------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) model :-   PCA object
    2) path :- Path of the model.
    ---------------------------------------------------------------------
    returns :- None
    ---------------------------------------------------------------------
    """
    pickle.dump(model, open(path, "wb"))


def get_optimial_number_of_features_count(
    data: pd.DataFrame,
    is_scaled: bool = True,
    threshold_Percentage_Of_Variation: float = 0.95,
) -> int:
    """
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    description : This Function will take data and few additional arguments, compute the optimial number of features that can be generated for following data and return that number
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) data :-   This should be pandas dataframe which should be cleaned and contain numeric values.
    2) is_scaled :- If data is already scaled, please set this parameter as true else set this as false
    3) threshold_Percentage_Of_Variation :- The maximum amount of threshold value of variance which programmer wants into his new features from existing data.
    ---------------------------------------------------------------------
    returns :- number of optimal features for the following data (int)
    ---------------------------------------------------------------------
    """
    if is_scaled == False:
        data: pd.DataFrame = data.apply(zscore)
    pca_model: PCA = fit_pca_model(data, data.shape[1])
    cumulative_variance: np.array = np.cumsum(pca_model.explained_variance_ratio_)
    return cumulative_variance[
        cumulative_variance <= threshold_Percentage_Of_Variation
    ].shape[0]


def feature_generation_by_pca(
    data: pd.DataFrame,
    target_column: str,
    is_scaled: bool = True,
    threshold_Percentage_Of_Variation: float = 0.95,
    load_Pre_trained_model: bool = False,
    pre_trained_model_Path: str = None,
) -> np.array:
    """
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    description : This function take data with some other arguments, fit or load pretrain mode according to the arguments specified and tranform the data from n dimentions to m dimensions
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    arguments :
    1) data :-   This should be pandas dataframe which should be cleaned and contain numeric values.
    2) target_column :- target column  name
    2) is_scaled :- If data is already scaled, please set this parameter as true else set this as false
    3) threshold_Percentage_Of_Variation :- The maximum amount of threshold value of variance which programmer wants into his new features from existing data.
    4) load_Pre_trained_model :- If programmer wants to use pre trained mode for transformation assign True, else False
    5) pre_trained_model_Path :- path of the pretrained mode from where we need to load the weights.
    ---------------------------------------------------------------------
    returns :- features in numpy array form
    ---------------------------------------------------------------------
    """
    target_column_values: pd.Series = data[target_column]
    # extracting numeric column name
    numeric_columns, categorical_columns = extract_numeric_and_categorical_column_names(
        data
    )

    non_numeric_data = data[categorical_columns]
    numeric_data = data[numeric_columns]

    if target_column in numeric_columns:
        numeric_data: pd.DataFrame = numeric_data.drop([target_column], axis=1)

    if is_scaled == False:
        numeric_data: pd.DataFrame = numeric_data.apply(zscore)
    # getting optimial number of features.
    optimial_number_of_features_count: int = get_optimial_number_of_features_count(
        data=numeric_data,
        is_scaled=is_scaled,
        threshold_Percentage_Of_Variation=threshold_Percentage_Of_Variation,
    )
    # loading the pretrained model
    if load_Pre_trained_model:
        pca_model: PCA = load_pca_model(pre_trained_model_Path)
    # fitting the new mode on the data.
    else:
        pca_model: PCA = fit_pca_model(numeric_data, optimial_number_of_features_count)
        save_pca_model(pca_model, os.path.join(os.getcwd(), "models", "pca_model.sav"))

    numeric_data: pd.DataFrame = pd.DataFrame(pca_model.transform(numeric_data))

    if target_column in numeric_columns:
        numeric_data[target_column] = target_column_values
    data = pd.concat([non_numeric_data, numeric_data], axis=1, join="inner")
    return data


"""
resampling techs
"""
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE


def adasyn():
    return ADASYN()


def smote():
    return SMOTE()


def resampling_fit_transform(
    data: pd.DataFrame, target_variable: str, resampliiopng_method
) -> pd.DataFrame:
    resampling_obj = resampling_method()
    predictor_variables, target_var = resampling_obj.fit_resample(
        data.drop([target_variable], axis=1, inplace=False), data[target_variable]
    )
    print("\n\n")
    target_var = np.array(target_var)
    return pd.DataFrame(
        np.append(predictor_variables, target_var.reshape(len(target_var), 1), axis=1),
        columns=data.columns,
    )


# 3. FUNCTION USED IN MODELLING AND REGRESSION
# NEEDED


def train_and_test_split(df, target_variable):
    """
    Splitting Data to two sets: Train and Test
    Input Parameters:
                     a. DataFrame
                     b. Target Variable: 'target_variable'
    Returns: train dataset, test dataset, train target, test target
    """
    columns = list(df.columns)
    columns.remove(target_variable)
    X = df[columns]
    # X = pd.get_dummies(X)
    Y = df[target_variable]
    # stratify is required for classification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1, stratify=Y
        )
    #  stratify is not required for regression
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1
        )
    return X_train, X_test, y_train, y_test


# 4. CLASSIFICATION MODELLING

# 4.1
import os
import pickle
import datetime

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


# performance metrics to observe in GridSearchCV
def scoring(regr):
    if regr:
        return ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]
    else:
        return ["accuracy", "f1", "precision", "recall"]


# Model fitting
def model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr):
    # if gridsearchcv is true then return the best fit model from gcv
    if gcv:
        metrics = []
        metrics = scoring(regr=regr)
        model = GCV(
            estimator=model,
            param_grid=gcv_params,
            cv=cv_indices,
            scoring=metrics,
            refit=metrics[1],
            n_jobs=-1,
            verbose=4,
        )

    model = model.fit(X_train, y_train)
    return model


##logistic Regression
def logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        C = gcv_params.get("C")[0]
        penalty = gcv_params.get("penalty")[0]
        tol = gcv_params.get("tol")[0]

        model = (
            LogisticRegression(
                solver="liblinear", C=C, penalty=penalty, tol=tol, random_state=0
            )
            if len(set(y_train)) == 2
            else LogisticRegression(
                multi_class="multinomial", C=C, penalty=penalty, tol=tol, random_state=0
            )
        )
    else:
        model = (
            LogisticRegression(solver="liblinear", random_state=0)
            if len(set(y_train)) == 2
            else LogisticRegression(multi_class="multinomial", random_state=0)
        )

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# support vector classifier
def support_vector_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        kernel = gcv_params.get("kernel")[0]
        C = gcv_params.get("C")[0]

        if kernel == "poly":
            degree = gcv_params.get("degree")[0]
        elif kernel == "rbf":
            gamma = gcv_params.get("gamma")[0]
        else:
            degree = 1
            gamma = "scale"

        model = SVC(
            probability=True,
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            random_state=0,
        )
    else:
        model = SVC(probability=True, random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# decision trees
def decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        max_depth = gcv_params.get("max_depth")[0]
        min_samples_leaf = gcv_params.get("min_samples_leaf")[0]
        min_samples_split = gcv_params.get("min_samples_split")[0]
        criterion = gcv_params.get("criterion")[0]

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=0,
        )
    else:
        model = DecisionTreeClassifier(random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


##Random forest
def random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        max_depth = gcv_params.get("max_depth")[0]
        min_samples_leaf = gcv_params.get("min_samples_leaf")[0]
        min_samples_split = gcv_params.get("min_samples_split")[0]
        criterion = gcv_params.get("criterion")[0]
        n_estimators = gcv_params.get("n_estimators")[0]

        model = RandomForestClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            n_estimators=n_estimators,
            random_state=0,
        )
    else:
        model = RandomForestClassifier(random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


## KNN
def knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        n_neighbors = gcv_params.get("n_neighbors")[0]
        weights = gcv_params.get("weights")[0]

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    else:
        model = KNeighborsClassifier()

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


## Xgboost
def xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        eta = gcv_params.get("eta")[0]
        gamma = gcv_params.get("gamma")[0]
        max_depth = gcv_params.get("max_depth")[0]

        model = XGBClassifier(eta=eta, gamma=gamma, max_depth=max_depth, random_state=0)
    else:
        model = XGBClassifier(random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


## Adaboost Classifier
def adaboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        n_estimators = gcv_params.get("n_estimators")[0]
        lr = gcv_params.get("learning_rate")[0]

        model = AdaBoostClassifier(
            n_estimators=n_estimators, learning_rate=lr, random_state=0
        )
    else:
        model = AdaBoostClassifier(random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# naive bayes classifier model
def naive_bayes_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        var_smoothing = gcv_params.get("var_smoothing")[0]
        model = GaussianNB(var_smoothing=var_smoothing)
    else:
        model = GaussianNB()

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


## Multinomial NaiveBayes
def multinomailNB_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        alpha = gcv_params.get("alpha")[0]
        model = MultinomialNB(alpha=alpha)
    else:
        model = MultinomialNB()

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# Multi Layer perceptron classifier model
def multi_layer_perceptron_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction class labels.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        hidden_layer_sizes = gcv_params.get("hidden_layer_sizes")[0]
        activation = gcv_params.get("activation")[0]
        solver = gcv_params.get("solver")[0]
        alpha = gcv_params.get("alpha")[0]
        lr = gcv_params.get("learning_rate")[0]

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=lr,
            random_state=0,
        )
    else:
        model = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0
        )

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# s3 bucket config
def save_model_to_s3(file_name, model):
    import io
    import boto3
    from decouple import config

    access_key = config("AWS_ACCESS_KEY_ID")
    secret_key = config("AWS_SECRET_ACCESS_KEY")
    region = config("AWS_REGION")
    bucket = config("AWS_S3_BUCKET_NAME")

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3 = session.resource("s3")

    root = "pickels/"
    # file_name = "sample_pickle.pkl"
    path = root + file_name + ".pkl"
    pickled_model = pickle.dumps(model)  # dump didnt work
    s3.Bucket(bucket).put_object(Key=path, Body=pickled_model)
    location = session.client("s3").get_bucket_location(Bucket=bucket)[
        "LocationConstraint"
    ]
    uploaded_url = f"https://s3-{location}.amazonaws.com/{bucket}/{path}"
    return uploaded_url


# model evaluation for classification
def model_evaluation_for_classification(
    df: pd.DataFrame,
    target_variable: str,
    output_file_name: str,
    fit_model,
    custom_selection,
    cv,
    cross_validation_,
    cv_params,
    gcv,
    gcv_params,
):
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in target column.
    df[target_variable] = label_encoder.fit_transform(df[target_variable])
    # performing the train test split
    X_train, X_test, y_train, y_test = train_and_test_split(df, target_variable)

    y_train = np.array(y_train)
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()

    result = None
    if cv:
        # cv=T, gcv = T/F
        result, model = cross_validation_(
            X_train,
            y_train,
            fit_model,
            cv_params,
            custom_selection,
            gcv,
            gcv_params,
            regr=False,
        )
    else:
        # if gcv=T,cv=F - by default gcv will take stratified kfold cv
        if gcv:
            model = fit_model(
                X_train, y_train, 10, custom_selection, gcv, gcv_params, regr=False
            )
            result = pd.DataFrame.from_dict(model.cv_results_)
            result = result.loc[result["mean_test_f1"] == result["mean_test_f1"].max()][
                [
                    "params",
                    "mean_test_accuracy",
                    "mean_test_precision",
                    "mean_test_recall",
                    "mean_test_f1",
                ]
            ]
            # print(result, model)
        else:
            # if cv=F, gcv=F. Fit the model
            model = fit_model(
                X_train, y_train, 10, custom_selection, gcv, gcv_params, regr=False
            )
    # getting label order
    labels_order: list = y_test.unique()
    # making the predictions
    test_pred: np.array = model.predict(X_test)
    test_pred_probs: np.array = model.predict_proba(X_test)
    # getting probs value for y true
    test_pred_prob: np.array = np.array(
        [array[test_pred[index]] for index, array in enumerate(test_pred_probs)]
    )
    # ROC curve
    false_positive_rate = dict()
    true_positive_rate = dict()
    thresholds = dict()
    auc_score = dict()
    for category in labels_order:
        original_label = label_encoder.inverse_transform([category])[0]
        (
            false_positive_rate[int(original_label)],
            true_positive_rate[(int(original_label))],
            thresholds[int(original_label)],
        ) = roc_curve(
            y_test, test_pred_prob, pos_label=category, drop_intermediate=False
        )
        auc_score[original_label] = auc(
            false_positive_rate[int(original_label)],
            true_positive_rate[int(original_label)],
        )
    # getting confusion matrix
    confusion_matrix_result = confusion_matrix(y_test, test_pred, labels=labels_order)
    # computing the classification report
    classification_report_result = (
        pd.DataFrame(
            classification_report(y_test, test_pred, output_dict=True, digits=2)
        )
        .transpose()
        .round(2)
    )
    # computing the accuracy score
    accuracy_score = metrics.accuracy_score(y_test, test_pred)
    # saving the model`
    if output_file_name != None:
        pickle_url = save_model_to_s3(output_file_name, model)
    labels_order = label_encoder.inverse_transform(labels_order)
    model_status = "Completed training"
    X_test["actual_output"] = label_encoder.inverse_transform(y_test)
    X_test["predicted_output"] = label_encoder.inverse_transform(test_pred)
    return (
        confusion_matrix_result,
        dict(classification_report_result.mean()),
        classification_report_result.reset_index(),
        accuracy_score,
        labels_order,
        false_positive_rate,
        true_positive_rate,
        thresholds,
        auc_score,
        X_test,
        str(datetime.datetime.now()),
        model_status,
        pickle_url,
        result,
    )


"""
Regression tasks functions
"""
# Linear regression
def linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):

    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    model = LinearRegression()
    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# #support vector machine
def support_vector_machine_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        kernel = gcv_params.get("kernel")[0]
        C = gcv_params.get("C")[0]

        if kernel == "poly":
            degree = gcv_params.get("degree")[0]
        elif kernel == "rbf":
            gamma = gcv_params.get("gamma")[0]
        else:
            degree = 1
            gamma = "scale"

        model = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)
    else:
        model = SVR()

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# #decision tree regression
def decision_tree_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        max_depth = gcv_params.get("max_depth")[0]
        min_samples_leaf = gcv_params.get("min_samples_leaf")[0]
        min_samples_split = gcv_params.get("min_samples_split")[0]
        criterion = gcv_params.get("criterion")[0]

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=0,
        )
    else:
        model = DecisionTreeRegressor(random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# Random Forest regression
def random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        max_depth = gcv_params.get("max_depth")[0]
        min_samples_leaf = gcv_params.get("min_samples_leaf")[0]
        min_samples_split = gcv_params.get("min_samples_split")[0]
        criterion = gcv_params.get("criterion")[0]
        n_estimators = gcv_params.get("n_estimators")[0]

        model = RandomForestRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            n_estimators=n_estimators,
            random_state=0,
        )
    else:
        model = RandomForestRegressor(random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# Xgb_regressor
def xgb_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        eta = gcv_params.get("eta")[0]
        gamma = gcv_params.get("gamma")[0]
        max_depth = gcv_params.get("max_depth")[0]

        model = xg.XGBRegressor(
            objective="reg:squarederror", eta=eta, gamma=gamma, max_depth=max_depth
        )
    else:
        model = xg.XGBRegressor(objective="reg:squarederror")

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# k-nearest neighbour regressor model
def kneighbors_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        n_neighbors = gcv_params.get("n_neighbors")[0]
        weights = gcv_params.get("weights")[0]

        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    else:
        model = KNeighborsRegressor()

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# ploynomial regression
def PolynomialRegression(degree=2):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())


def polynomial_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. DataFrame
                    b. Target Variable: 'target_variable'
                    c. Output File Name: 'output_file_name' or None (If don't want to save)
    Returns:
            Returns root mean squared error value for model evaluation
    """
    # splitting the data into train and test sets.
    # fitting polynomial regression to the dataset
    if gcv:
        model = PolynomialRegression()
        poly_X = X_train
    else:
        model = PolynomialFeatures(
            degree=gcv_params.get("polynomialfeatures__degree")[0]
        )
        poly_X = model.fit_transform(X_train)
        model = LinearRegression()

    return model_computation(poly_X, y_train, model, cv_indices, gcv, gcv_params, regr)


# lasso regression
def lasso_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        alpha = gcv_params.get("alpha")[0]
        model = Lasso(alpha=alpha)
    else:
        model = Lasso()

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# ridge regression
def ridge_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        alpha = gcv_params.get("alpha")[0]
        model = Ridge(alpha=alpha)
    else:
        model = Ridge(alpha=0.1)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# ElasticNet regression
def elasticnet_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        alpha = gcv_params.get("alpha")[0]
        l1_ratio = gcv_params.get("l1_ratio")[0]

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
    else:
        model = ElasticNet(alpha=0.1, random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# stochastic gradient descent classifier model
def sgd_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        alpha = gcv_params.get("alpha")[0]
        l1_ratio = gcv_params.get("l1_ratio")[0]
        penalty = gcv_params.get("penalty")[0]
        lr = gcv_params.get("learning_rate")[0]

        model = SGDRegressor(
            alpha=alpha,
            l1_ratio=l1_ratio,
            penalty=penalty,
            learning_rate=lr,
            random_state=0,
        )
    else:
        model = SGDRegressor(alpha=0.1, random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# Gradient boosting regression model
def gradientboosting_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_indices,
    custom_selection,
    gcv,
    gcv_params,
    regr,
):
    """
    Create and train machine learning model for prediction of independent variable
    based on dependent variables in the dataframe.
    Input Parameters:
                    a. X_train : dataframe
                    b. y_train : pd.series
                    c. cv_indices: the selected cv train,val splits.
                    d. gcv: True/False
                    e. gcv_params: selected hyper params for gridsearchcv if gcv=true
                    f. regr = If regression true/false
    Returns:
            Returns fitted model
    """
    if custom_selection:
        max_depth = gcv_params.get("max_depth")[0]
        min_samples_leaf = gcv_params.get("min_samples_leaf")[0]
        min_samples_split = gcv_params.get("min_samples_split")[0]
        criterion = gcv_params.get("criterion")[0]
        n_estimators = gcv_params.get("n_estimators")[0]

        model = GradientBoostingRegressor(
            alpha=0.1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            n_estimators=n_estimators,
            random_state=0,
        )
    else:
        model = GradientBoostingRegressor(alpha=0.1, random_state=0)

    return model_computation(X_train, y_train, model, cv_indices, gcv, gcv_params, regr)


# light gradient boosting machine regression model


# def lgbm_regression(df, target_variable, output_file_name):
#     """
#     Create and train machine learning model for prediction of independent variable
#     based on dependent variables in the dataframe.
#     Input Parameters:
#                     a. DataFrame
#                     b. Target Variable: 'target_variable'
#                     c. Output File Name: 'output_file_name' or None (If don't want to save)
#     Returns:
#             Returns root mean squared error value for model evaluation
#     """
#     X_train, X_test, y_train, y_test = train_and_test_split(df, target_variable)
#     model = LGBMRegressor(random_state=0)
#     model = model.fit(X_train, y_train)
#     test_pred = model.predict(X_test)
#     X_test["actual_output"] = y_test
#     X_test["predicted_output"] = test_pred
#     rmse_score = math.sqrt(mean_squared_error(y_test, test_pred))
#     if output_file_name != None:
#         pickle.dump(model, open(output_file_name + ".sav", "wb"))
#         print(
#             "Model Weights:",
#             output_file_name + ".sav",
#             ", got saved to location:",
#             os.getcwd() + "\" + output_file_name + ".sav",
#         )
#     return rmse_score, X_test


# # catboost regression model


# def catboost_regression(df, target_variable, output_file_name):
#     """
#     Create and train machine learning model for prediction of independent variable
#     based on dependent variables in the dataframe.
#     Input Parameters:
#                     a. DataFrame
#                     b. Target Variable: 'target_variable'
#                     c. Output File Name: 'output_file_name' or None (If don't want to save)
#     Returns:
#             Returns root mean squared error value for model evaluation
#     """
#     X_train, X_test, y_train, y_test = train_and_test_split(df, target_variable)
#     model = CatBoostRegressor(random_state=0)
#     model = model.fit(X_train, y_train)
#     test_pred = model.predict(X_test)
#     X_test["actual_output"] = y_test
#     X_test["predicted_output"] = test_pred
#     rmse_score = math.sqrt(mean_squared_error(y_test, test_pred))
#     if output_file_name != None:
#         pickle.dump(model, open(output_file_name + ".sav", "wb"))
#         print(
#             "Model Weights:",
#             output_file_name + ".sav",
#             ", got saved to location:",
#             os.getcwd() + "\" + output_file_name + ".sav",
#         )
#     return rmse_score, X_test


## Model evaluation for regression
def model_evaluation_for_regression(
    df: pd.DataFrame,
    target_variable: str,
    output_file_name: str,
    fit_model,
    custom_selection,
    cv,
    cross_validation_,
    cv_params,
    gcv,
    gcv_params,
):
    # split train and test sets
    X_train, X_test, y_train, y_test = train_and_test_split(df, target_variable)

    y_train = np.array(y_train)
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()

    result = None
    if cv:
        # cv=T, gcv=T/F
        result, model = cross_validation_(
            X_train,
            y_train,
            fit_model,
            cv_params,
            custom_selection,
            gcv,
            gcv_params,
            regr=True,
        )

        if cross_validation_ == nested_cross_validation:
            result = pd.DataFrame.from_dict(model.cv_results_)
            result = df_result.loc[
                result["mean_test_neg_root_mean_squared_error"]
                == result["mean_test_neg_root_mean_squared_error"].min()
            ][
                [
                    "params",
                    "mean_test_neg_root_mean_squared_error",
                    "mean_test_neg_mean_absolute_error",
                    "mean_test_r2",
                ]
            ]
    else:
        # if gcv=T,cv=F - by default gcv will take stratified kfold cv
        if gcv:
            model = fit_model(
                X_train, y_train, 10, custom_selection, gcv, gcv_params, regr=True
            )
            result = pd.DataFrame.from_dict(model.cv_results_)
            result = result.loc[
                result["mean_test_neg_root_mean_squared_error"]
                == result["mean_test_neg_root_mean_squared_error"].min()
            ][
                [
                    "params",
                    "mean_test_neg_root_mean_squared_error",
                    "mean_test_neg_mean_absolute_error",
                    "mean_test_r2",
                ]
            ]
            # print(result,model)
        # cv=F, gcv=F, then fit the model
        else:
            model = fit_model(
                X_train, y_train, 10, custom_selection, gcv, gcv_params, regr=True
            )

    # prediction on test data
    # if polynomial regression, we need to tranform x_test
    if fit_model == polynomial_regression:
        if gcv:
            degree = result.params.values[0].get("polynomialfeatures__degree")
        else:
            degree = gcv_params.get("polynomialfeatures__degree")[0]

        poly_feats = PolynomialFeatures(degree=degree)
        poly_X_train = poly_feats.fit_transform(X_train)

        lr = LinearRegression()
        lr.fit(poly_X_train, y_train)

        test_pred = poly_feats.transform(X_test)
        test_pred = lr.predict(test_pred)
    else:
        test_pred = model.predict(X_test)

    rmse_score = math.sqrt(mean_squared_error(y_test, test_pred))
    X_test["actual_output"] = y_test
    X_test["predicted_output"] = test_pred
    if output_file_name != None:
        pickle_url = save_model_to_s3(output_file_name, model)
    numeric_columns, _ = extract_numeric_and_categorical_column_names(X_test)
    X_test[numeric_columns] = X_test[numeric_columns].apply(
        lambda column: pd.Series.round(column, 2)
    )

    return (rmse_score, X_test, pickle_url, result)


# 6. PREDICTION


def load_from_s3(uploaded_url):
    byteurl = requests.get(uploaded_url, stream=True)
    weights_buffer = io.BytesIO(byteurl.content)
    pickled_model = pickle.load(weights_buffer)
    return pickled_model


def filter_preprocess_steps(meta_data):
    all_procesess = []
    if meta_data["skipped"] == False:
        pre_process = meta_data["pre_processing"]
        for process in pre_process:
            if pre_process[process] == True:
                all_procesess.append(process)
    else:
        pass
    return all_procesess


def apply_preprocess_steps(all_procesess, dataset, metadata):
    if "data_type_correction" in all_procesess:
        dataset = data_type_correction(dataset)
    if "auto_remove_unwanted_columns" in all_procesess:
        dataset = auto_remove_unwanted_columns(dataset)
    if "remove_correlated_columns" in all_procesess:
        dataset = remove_correlated_columns(
            dataset, threshold=metadata["pre_processing"]["threshold_value"]
        )
    return dataset


def predict(metadata, model_weights_pkl, scaler_pkl, input_array):
    """
    Prediction of output based on user input provided to model
    Input Parameters:
                    a. metadata with preprocessing steps information
                    b. Model Weights_pkl: saved model weight file (.sav pickel file )
                    c. scaler_pkl: pickle file saved at run time
                    c. Input Array: input array ( dependent feature attributes)
                       example: [1,2,1,2]
    Returns:
           Returns prediction results
    """
    all_procesess = filter_preprocess_steps(metadata)
    if len(all_procesess) != 0:
        dataset = apply_preprocess_steps(all_procesess, input_array, metadata)
        if "onehotencoding" in all_procesess:
            dataset = categorical_value_encoder_transformer(dataset, one_hot_encoding)
        if "standard_scale" or "min_max_scale" or "robust_scale" in all_procesess:
            scaler = load_from_s3(scaler_pkl)
            print(f"this pickle file have ({scaler.n_features_in_}) features")
            dataset = scaler.transform(dataset)
        loaded_model = load_from_s3(model_weights_pkl)
        predicted = loaded_model.predict(dataset)
        # label_encoder
        label_map = metadata["pre_processing"]["label_encoder"]
        if label_map is not None:
            label_map = {val: key for key, val in label_map.items()}
            predicted = [label_map[prediction] for prediction in predicted]
        return predicted
    else:
        dataset = input_array
        loaded_model = load_from_s3(model_weights_pkl)
        predicted = loaded_model.predict(dataset)
        # label_map = metadata["pre_processing"]["label_encoder"]
        # if label_map is not None:
        #     label_map = {val: key for key, val in label_map.items()}
        #     predicted = [label_map[prediction] for prediction in predicted]
        return predicted


# time series
def preprocess(dataset):
    for i in dataset.columns:
        if dataset[i].dtype == "O":
            dataset[i] = pd.to_datetime(dataset[i])
            dataset = dataset.set_index(i)
        elif dataset[i].dtype == "<M8[ns]":
            dataset = dataset.set_index(i)
        else:
            pass
    return dataset


def auto_arima_model(dataset, date_column, target_column, seasonality: bool):
    dataset = dataset[[date_column, target_column]]
    dataset = preprocess(dataset)
    dataset = dataset.sort_index()
    # splitting the data into train and test
    sz = int(len(dataset) * 0.8)
    train_data = dataset[:sz]
    test_data = dataset[sz:]
    output_filename = f"auto_ariama_{uuid.uuid4().hex}"
    # apply the auto arima model
    # we are try with the p,d,q values ranging from 0 to 5 to get better optimal values from the models
    if seasonality == True:
        model = auto_arima(
            train_data,
            start_p=0,
            d=1,
            start_q=0,
            max_p=5,
            max_d=5,
            max_q=5,
            start_P=0,
            D=1,
            start_Q=0,
            max_P=5,
            max_D=5,
            max_Q=5,
            seasonal=True,
        )
        #     model = auto_arima(train_data, trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(train_data)
    else:
        model = auto_arima(
            train_data,
            start_p=0,
            d=1,
            start_q=0,
            max_p=5,
            max_d=5,
            max_q=5,
            start_P=0,
            D=1,
            start_Q=0,
            max_P=5,
            max_D=5,
            max_Q=5,
            seasonal=False,
        )
        model.fit(train_data)
    forecast = model.predict(n_periods=len(test_data))
    forecast = pd.DataFrame(forecast, index=test_data.index, columns=["Prediction"])
    forecast["Actual_data"] = test_data
    error = mean_squared_error(test_data, forecast["Prediction"])
    rmse = np.sqrt(error)
    if output_filename != None:
        pickle_url = save_model_to_s3(output_filename, model)
    return forecast, rmse, pickle_url


def time_series_predict(data, pickle_url):
    model = load_from_s3(pickle_url)
    prediction = model.predict(n_periods=len(data))
    forecast = pd.DataFrame()
    forecast["prediction"] = prediction
    # forecast["Date"] = data["Date"]
    return forecast


############################################################################################
##updated on July'22

# metrics to calcualte cross validation scores.
def cross_validation_evaluation(
    X_, y_, fit_model, cv_type, custom_selection, gcv, gcv_params, regr
):

    ##for regression models
    if regr:
        if gcv:
            model = fit_model(X_, y_, cv_type, custom_selection, gcv, gcv_params, regr)
            result = pd.DataFrame.from_dict(model.cv_results_)
            metrics = result.loc[
                result["mean_test_neg_root_mean_squared_error"]
                == result["mean_test_neg_root_mean_squared_error"].min()
            ][
                [
                    "params",
                    "mean_test_neg_root_mean_squared_error",
                    "mean_test_neg_mean_absolute_error",
                    "mean_test_r2",
                ]
            ]

        else:
            scoring = {
                "rmse": make_scorer(mean_squared_error),
                "mae": make_scorer(mean_absolute_error),
                "r2": make_scorer(r2_score),
            }
            model = fit_model(X_, y_, 10, custom_selection, gcv, gcv_params, regr)
            metrics = pd.DataFrame(
                cross_validate(model, X_, y_, cv=cv_type, scoring=scoring)
            )

    ##for classification models
    else:
        if gcv:
            model = fit_model(X_, y_, cv_type, custom_selection, gcv, gcv_params, regr)
            result = pd.DataFrame.from_dict(model.cv_results_)
            metrics = result.loc[
                result["mean_test_f1"] == result["mean_test_f1"].max()
            ][
                [
                    "params",
                    "mean_test_accuracy",
                    "mean_test_precision",
                    "mean_test_recall",
                    "mean_test_f1",
                ]
            ]
        else:
            scoring = {
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score),
                "recall": make_scorer(recall_score),
                "f1": make_scorer(f1_score),
            }
            model = fit_model(X_, y_, 10, custom_selection, gcv, gcv_params, regr)
            metrics = pd.DataFrame(
                cross_validate(model, X_, y_, cv=cv_type, scoring=scoring)
            )

    return metrics, model


# leavePout CV
def lp_out_cross_validation(
    X_, y_, fit_model, cv_params, custom_selection, gcv, gcv_params, regr
):
    p = cv_params.get("p")
    lpo = LeavePOut(p=p).split(X_)
    metrics, model = cross_validation_evaluation(
        X_, y_, fit_model, lpo, custom_selection, gcv, gcv_params, regr
    )

    return metrics, model


# leaveoneOut CV
def lo_out_cross_validation(
    X_, y_, fit_model, cv_params, custom_selection, gcv, gcv_params, regr
):
    loo = LeaveOneOut().split(X_)
    metrics, model = cross_validation_evaluation(
        X_, y_, fit_model, loo, custom_selection, gcv, gcv_params, regr
    )

    return metrics, model


# holdout cv
def hold_out_cross_validation(
    X_, y_, fit_model, cv_params, custom_selection, gcv, gcv_params, regr
):
    test_size = cv_params.get("test_size")
    ho = ShuffleSplit(n_splits=1, test_size=test_size, random_state=0).split(X_)
    metrics, model = cross_validation_evaluation(
        X_, y_, fit_model, ho, custom_selection, gcv, gcv_params, regr
    )

    return metrics, model


# shuffle split cv
def rep_rand_sample_cross_validation(
    X_, y_, fit_model, cv_params, custom_selection, gcv, gcv_params, regr
):
    n_splits = cv_params("n_splits")
    test_size = cv_params.get("test_size")
    ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0).split(X_)
    metrics, model = cross_validation_evaluation(
        X_, y_, fit_model, ss, custom_selection, gcv, gcv_params, regr
    )

    return metrics, model


# kfold cv
def k_fold_cross_validation(
    X_, y_, fit_model, cv_params, custom_selection, gcv, gcv_params, regr
):
    n_splits = cv_params.get("n_splits")
    k_fold = KFold(n_splits=n_splits).split(X_)
    metrics, model = cross_validation_evaluation(
        X_, y_, fit_model, k_fold, custom_selection, gcv, gcv_params, regr
    )

    return metrics, model


# startified kfold cv
def k_fold_stratified_cross_validation(
    X_, y_, fit_model, cv_params, custom_selection, gcv, gcv_params, regr
):
    n_splits = cv_params.get("n_splits")
    skf = StratifiedKFold(n_splits=n_splits).split(X_, y_)
    metrics, model = cross_validation_evaluation(
        X_, y_, fit_model, skf, custom_selection, gcv, gcv_params, regr
    )

    return metrics, model


# nested cv
def nested_cross_validation(
    X_, y_, fit_model, cv_params, custom_selection, gcv, gcv_params, regr
):
    df_metrics = pd.DataFrame()

    n_splits = cv_params.get("n_splits_outer")
    inner_cv = KFold(n_splits=n_splits)

    n_splits = cv_params.get("n_splits_inner")
    outer_cv = KFold(n_splits=n_splits)

    rmse = []
    mae = []
    r2 = []
    accuracy = []
    f1 = []
    precision = []
    recall = []

    # Nested CV with parameter optimization
    nestedscore = []
    for train_outer, test_outer in outer_cv.split(X_):
        X_train_outer, X_test_outer, y_train_outer, y_test_outer = (
            X_.loc[train_outer],
            X_.loc[test_outer],
            y_[train_outer],
            y_[test_outer],
        )
        clf = fit_model(
            X_train_outer,
            y_train_outer,
            inner_cv.split(X_train_outer),
            custom_selection,
            True,
            gcv_params,
            regr,
        )
        y_pred_outer = clf.predict(X_test_outer)

        if regr:
            rmse.append(np.round(mean_squared_error(y_test_outer, y_pred_outer), 4))
            mae.append(np.round(mean_absolute_error(y_test_outer, y_pred_outer), 4))
            r2.append(np.round(r2_score(y_test_outer, y_pred_outer), 4))

            mean = dict({"rmse": np.mean(rmse), "mae": np.mean(mae), "r2": np.mean(r2)})
            metrics = pd.DataFrame({"rmse": rmse, "mae": mae, "r2": r2})
        else:
            accuracy.append(np.round(accuracy_score(y_test_outer, y_pred_outer), 4))
            f1.append(np.round(f1_score(y_test_outer, y_pred_outer), 4))
            precision.append(np.round(precision_score(y_test_outer, y_pred_outer), 4))
            recall.append(np.round(recall_score(y_test_outer, y_pred_outer), 4))

            mean = dict(
                {
                    "accuracy": np.mean(accuracy),
                    "precision": np.mean(precision),
                    "recall": np.mean(recall),
                    "f1-score": np.mean(f1),
                }
            )
            metrics = pd.DataFrame(
                {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1,
                }
            )
        df_metrics = df_metrics.append(metrics).reset_index(drop=True)

    return df_metrics, clf
