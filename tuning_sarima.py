import pandas as pd
import statsmodels.api as sm
import itertools
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
import numpy as np


symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD','BCHUSD','XRPUSD','LINKUSD','ADAUSD','DOTUSD','UNIUSD','DOGEUSD','ETCUSD','MATICUSD',
    'BSVUSD','FILUSD','ATOMUSD','XLMUSD','AAVEUSD','CAKEUSD','SUSHIUSD','MKRUSD','AVAXUSD']

PDQ_DICT = {}

for smb in symbols:
    df = pd.read_csv(f'historical_data/{smb}.csv')

    model_data = df.copy()
    model_data.index = model_data['date']
    model_data = model_data['close']


    def objective(trial, model_data=model_data):
        p = d = q = range(0, 3)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 4) for x in list(itertools.product(p, d, q))]
        model = sm.tsa.statespace.SARIMAX(model_data, 
                                        order=trial.suggest_categorical("pdq",pdq),
                                        seasonal_order=trial.suggest_categorical("seasonal_pdq",seasonal_pdq)
                                        )
        #
        results = model.fit()
        pred = results.predict(0, len(model_data)+4)
        pred = pd.DataFrame(pred)
        #forecast_pred = pred.tail(5)

        #r2 = r2_score(pred[:-5], model_data).round(2)
        rmse = np.sqrt(mean_squared_error(pred[:-5], model_data)).round(2)
        #mae = mean_absolute_error(pred[:-5], model_data).round(2)
        return rmse

    study = optuna.create_study(direction='minimize')

    study.optimize(objective, n_trials=50)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    PDQ_DICT[smb] = study.best_trial.params

print(PDQ_DICT)