import numpy as np 
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split






def preprocessesing_pipeline(df:pd.DataFrame )->np.ndarray:
    #prepare data for ml
    """
        :param df: input dataFrame
        :param test_size: the way of spliting the Dataframe
        :return: the dataframe in four arrays 
    """
    df_1=df.copy()
    y=df_1['price']
    df_1.drop(labels='price',axis=1,inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(df_1, y, test_size=0.40,random_state=42)
    return x_train, x_test, y_train, y_test




def remove_outliers(x_train:np.ndarray, y_train:np.ndarray, y_test:np.ndarray)-> np.ndarray:
    """
        Removes high values that may impact the training of the model negatively
        
        :param x_test: test feature values array
        :param x_train: training feature valeus array
        :param y_test: test feature array
    """
    x_train=x_train[y_train<800]
    y_train=y_train[y_train<800]
    y_train=np.log(y_train)
    y_test=np.log(y_test)
    return x_train, y_train, y_test



def imputation(x_test:np.ndarray , x_train:np.ndarray)->np.ndarray:
    """
       :param x_test: test feature values array
       :param x_train: training feature values array
       :return: training and test feature array after imputation
    """
    imp = IterativeImputer(max_iter=20, random_state=0)
    imp.fit(x_train)
    x_train= imp.transform(x_train)
    x_test= imp.transform(x_test)
    return x_test,x_train



#scale training and test feature

def Scaling(x_test:np.ndarray , x_train:np.ndarray)->np.ndarray:
    """
        :param x_test: test feature values array
        :param x_train: training feature values array
        :return: training and test feature array after scaling
    """
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)
    return x_test ,x_train

def Splitting(df:pd.DataFrame)->np.ndarray:
    """
        :param df: input dataFrame
        :return : a copy of dataFrame prepared for fitting the model
    """
    x_train,x_test,y_train,y_test = preprocessesing_pipeline(df)
    [x_train, y_train, y_test] = remove_outliers(x_train, y_train, y_test)
    [x_test,x_train] = imputation(x_test,x_train)
    [x_test,x_train] = Scaling(x_test,x_train)
    return x_train, x_test, y_train, y_test
