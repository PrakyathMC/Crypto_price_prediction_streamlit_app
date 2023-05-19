import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly as plotly
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import feedparser
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import optuna
import keras


#symbols = ['LTCUSD','ATOMUSD','BCHUSD','LINKUSD']
symbol = 'ATOMUSD'
print(f"Starting {symbol}")
data = pd.read_csv(f'historical_data/{symbol}.csv')
data.reset_index(inplace=True)
data['date'] = pd.to_datetime(data['date']).dt.date
data = data.set_index('date')

num_of_days = 5



# MODEL TUNING DONE HERE"
def objective(trial, data=data, num_of_days=num_of_days):
    keras.backend.clear_session()
    n_cols = 1
    dataset = data[['close']]
    dataset = pd.DataFrame(dataset)
    dataset = dataset.values

    scaler = MinMaxScaler(feature_range= (0,1))
    scaled_data = scaler.fit_transform(np.array(dataset))
    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    print('Train Size:', train_size, 'Test Size:', test_size)
    #return train_size, test_size, scaled_data

    train_data = scaled_data[0:train_size, :]

    x_train = []
    y_train = []
    time_steps = 60

    for i in range(time_steps, len(train_data)):
        x_train.append(train_data[i - time_steps:i, :n_cols])
        y_train.append(train_data[i, :n_cols])

        # if i<=time_steps:
        #     print('X_train:', x_train)
        #     print('y_train:', y_train)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_cols))
    #return x_train, y_train
    
    # Tuning variables
    lstm_l1 = trial.suggest_int("lstm_units_L1", 32, 128, step=4)
    lstm_l2 = trial.suggest_int("lstm_units_L2", 64, 256, step=4)
    dense_l1 = trial.suggest_int("dense_units_L1", 32, 128, step=4)
    dense_l2 = trial.suggest_int("dense_units_L2", 16, 64, step=4)
    activation = trial.suggest_categorical("activation", ["tanh","relu", "selu", "elu",])
    dropout_rate = trial.suggest_float("lstm_dropout", 0.0, 0.3)
    # Tuning variables ends here

    #lstm model
    model = Sequential([
        LSTM(lstm_l1, return_sequences= True, input_shape=(x_train.shape[1], n_cols), activation=activation),
        LSTM(lstm_l2, return_sequences= False, activation=activation),
        Dense(dense_l1, activation=activation),
        Dense(dense_l2, activation=activation),
        Dropout(dropout_rate),
        Dense(n_cols)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
    #fitting lstm model to training set
    history = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)
    #return model.summary(),history.history

    #predictions
    test_data = scaled_data[train_size - time_steps:, :]

    x_test = []
    y_test = []

    for i in range(time_steps, len(test_data)):
        x_test.append(test_data[i - time_steps:i, 0:n_cols])
        y_test.append(test_data[i, 0:n_cols])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],n_cols))

    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)

    y_test = scaler.inverse_transform(y_test)

    #calculating evaluation metrics
    mse = np.mean((y_test - prediction)**2).round(2)
    #rmse = np.sqrt(np.mean((y_test - prediction)**2)).round(2)
    rmse = np.sqrt(mean_squared_error(y_test, prediction)).round(2)
    mae = mean_absolute_error(y_test, prediction).round(2)
    r2 = r2_score(y_test, prediction).round(2)
    preds_acts = pd.DataFrame(data={'Predictions': prediction.flatten(), 'Actuals': y_test.flatten()})
    #return rmse, preds_acts

    #future prediction
    last_day = data.index.max()
    future_dates = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=num_of_days, inclusive='left')

    future_preds = []
    last_time_step = scaled_data[-time_steps:]

    for i in range(num_of_days):
        input_data = last_time_step[-time_steps:]
        input_data = np.array(input_data)
        input_data = np.reshape(input_data, (1, time_steps, n_cols))

        predicted_prices = model.predict(input_data)
        future_preds.append(predicted_prices[0][0])

        last_time_step = np.append(last_time_step, predicted_prices, axis=0)
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
    future_prices_df = pd.DataFrame(future_preds, columns=['Predicted Close Price'], index=future_dates)
    #return history.history['loss'][-1],mse,mae,rmse,r2, preds_acts, future_prices_df
    return rmse


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=20)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

best_trial = study.best_trial.params
# MODEL TUNING DONE HERE"


# WITH best parameters the model will retrain and will be saved
print("Saving model....")
def model_save(data, num_of_days, best_trial):
    keras.backend.clear_session()
    n_cols = 1
    dataset = data[['close']]
    dataset = pd.DataFrame(dataset)
    dataset = dataset.values

    scaler = MinMaxScaler(feature_range= (0,1))
    scaled_data = scaler.fit_transform(np.array(dataset))
    train_size = int(len(dataset) * 0.80)
    test_size = len(dataset) - train_size
    print('Train Size:', train_size, 'Test Size:', test_size)
    #return train_size, test_size, scaled_data

    train_data = scaled_data[0:train_size, :]

    x_train = []
    y_train = []
    time_steps = 60

    for i in range(time_steps, len(train_data)):
        x_train.append(train_data[i - time_steps:i, :n_cols])
        y_train.append(train_data[i, :n_cols])


    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_cols))

    # Tuning variables
    lstm_l1 = best_trial["lstm_units_L1"]
    lstm_l2 = best_trial["lstm_units_L2"]
    dense_l1 = best_trial["dense_units_L1"]
    dense_l2 = best_trial["dense_units_L2"]
    activation = best_trial["activation"]
    dropout_rate = best_trial["lstm_dropout"]
    # Tuning variables ends here

    #lstm model
    model = Sequential([
        LSTM(lstm_l1, return_sequences= True, input_shape=(x_train.shape[1], n_cols), activation=activation),
        LSTM(lstm_l2, return_sequences= False, activation=activation),
        Dense(dense_l1, activation=activation),
        Dense(dense_l2, activation=activation),
        Dropout(dropout_rate),
        Dense(n_cols)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])

    #fitting lstm model to training set
    history = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)

    #predictions
    test_data = scaled_data[train_size - time_steps:, :]

    x_test = []
    y_test = []

    for i in range(time_steps, len(test_data)):
        x_test.append(test_data[i - time_steps:i, 0:n_cols])
        y_test.append(test_data[i, 0:n_cols])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],n_cols))

    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)

    y_test = scaler.inverse_transform(y_test)

    #calculating evaluation metrics
    mse = np.mean((y_test - prediction)**2).round(2)
    rmse = np.sqrt(np.mean((y_test - prediction)**2)).round(2)
    mae = mean_absolute_error(y_test, prediction).round(2)
    r2 = r2_score(y_test, prediction).round(2)
    preds_acts = pd.DataFrame(data={'Predictions': prediction.flatten(), 'Actuals': y_test.flatten()})

    #future prediction
    last_day = data.index.max()
    future_dates = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=num_of_days, inclusive='left')

    future_preds = []
    last_time_step = scaled_data[-time_steps:]

    for i in range(num_of_days):
        input_data = last_time_step[-time_steps:]
        input_data = np.array(input_data)
        input_data = np.reshape(input_data, (1, time_steps, n_cols))

        predicted_prices = model.predict(input_data)
        future_preds.append(predicted_prices[0][0])

        last_time_step = np.append(last_time_step, predicted_prices, axis=0)
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
    future_prices_df = pd.DataFrame(future_preds, columns=['Predicted Close Price'], index=future_dates)

    # Model Saving Here
    model.save(f"saved_model/{symbol}_model.keras", overwrite=True, save_format=None)
    print(r2, rmse, mae)
model_save(data, num_of_days, best_trial)

print(f"Ending {symbol}")