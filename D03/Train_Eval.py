import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,median_absolute_error
from sklearn.ensemble import RandomForestRegressor


def Train_Eval(x_test, x_train, y_test, y_train):
    """
        :param x_test: test feature values array
        :param x_train: training feature values array
        :param y_test: test feature array
        :param y_train: training feature array
        :return: the predictions on the testing set and the metrics(MAPE,medianAE,MAE)
    """
    pred = prediction(x_test, x_train, y_train)
    a = metrics(y_test,pred)
    return pred, a


def prediction(x_test:np.ndarray , x_train:np.ndarray, y_train:np.ndarray)-> np.ndarray:
    """
        :param x_test: test feature array
        :param x_train: training feature array
        :return: the predictions of the model
    """
    model=RandomForestRegressor()
    model.fit(x_train, y_train)
    predictions=model.predict(x_test)
    return predictions


def metrics(y_test:np.ndarray,predictions:np.ndarray)-> float :
    """
       :param y_test:test feature array
       :param predictions: the predictions of the model
       :return: the metrics(MAPE,medianAE,MAE)
    """
    y_test_=np.exp(y_test)
    predictions_=np.exp(predictions)
    metrics={'MAPE':{mean_absolute_percentage_error(y_test_, predictions_)},
             'MedianAE':{median_absolute_error(y_test_, predictions_)},
             'MAE':{mean_absolute_error(y_test_, predictions_)}
             }
    return metrics
