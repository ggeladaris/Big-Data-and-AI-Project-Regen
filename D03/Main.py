def main():
    import pandas as pd
    import numpy as np
    import re
    import ast
    from scipy import stats
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_absolute_percentage_error
    from sklearn.dummy import DummyRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from Preprocessing import Preprocessing
    from Splitting import Splitting
    from Train_Eval import Train_Eval



    df = pd.read_csv('listings.csv')
    df = Preprocessing(df)
    [x_train, x_test, y_train, y_test] = Splitting(df)
    [pred, metrics] = Train_Eval(x_test, x_train, y_test, y_train)
    pred = np.exp(pred)

    return print(pred, metrics)
main()


