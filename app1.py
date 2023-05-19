import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import datetime
import plotly as plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#import technical_analysis as ta
import feedparser
import datetime
import tensorflow as tf


from tingo_API import get_tiingo_data
import statsmodels.api as sm
from sarima_pdq import SARIMA_PDQ_DICT

st.title("SOLiGence Crypto Currency Price Prediction App")
st.markdown("<h2 style='color:green;'>About the Application</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size:20px;color:Orange;'>Welcome to our state-of-the-art interactive cryptocurrency forecasting tool, designed exclusively for SOLiGence. This application provides a unique platform for users to extract valuable insights from an array of cryptocurrency data and make informed conjectures regarding future price trends of select coins..</p>"
                "<p style='font-size:20px;color:red;'>Disclaimer: This application is designed to assist with data analysis and does not provide financial advice. It is vital to note that any forecast provided by the system does not guarantee future performance and must be used with caution.</p>", 
                unsafe_allow_html=True)
symbols = ['BTCUSD', 'ETHUSD', 'LTCUSD','BCHUSD','XRPUSD','LINKUSD','ADAUSD','DOTUSD','UNIUSD','DOGEUSD','ETCUSD','MATICUSD',
    'BSVUSD','FILUSD','ATOMUSD','XLMUSD','AAVEUSD','CAKEUSD','SUSHIUSD','MKRUSD','AVAXUSD']

def main():
    option = st.sidebar.radio("Make a choice",["DataFrame","Visualizations","Predict","Profit","Help"])
    if option == 'DataFrame':
        dataframe()
    elif option == 'Visualizations':
        visualization()
    elif option == 'Predict':
        predict()
    elif option == 'Profit':
        investment = st.number_input('How much money do you plan to invest?', min_value=0.0)
        profit_target = st.number_input('What is your profit target?', min_value=0.0)
        num_of_days = st.number_input('Over how many days?', min_value=1)
        if st.button('Calculate'):
            future_prices = predict_all(num_of_days)
            best_coin, max_profit = calculate_profit(investment, profit_target, num_of_days, future_prices)
            if best_coin is None:
                st.write('No investment will reach your profit target.')
            else:
                st.write(f'The best investment is {best_coin}, which is predicted to give a profit of {max_profit}.')
    else:
        display_help()
    

  
def load_data(symbol, start_date, end_date):
    #df = pdr.get_data_tiingo(symbol, start=start_date, end=end_date, api_key='74a8c7b7cc007d6c226cf707e5cace07704dee65')
    df = get_tiingo_data(symbol, start_date, end_date)

    # df.isnull().sum()
    # df.duplicated().sum()

    #df.drop(['adjClose','adjHigh','adjLow','adjOpen','adjVolume','divCash','splitFactor'], axis=1, inplace = True)
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.set_index('date')

    return df

choice = st.sidebar.selectbox("Select a coin", options=symbols)
choice = choice.upper()
today = datetime.date.today()
duration = st.sidebar.number_input("Enter the duration ", value = 365)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        load_data(choice, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = load_data(choice, start_date, end_date)
copy_data = data.copy()
scaler = StandardScaler()

def dataframe():
    st.write("This section provides insights to the data you have opted to use")
    st.dataframe(data.tail(10))
    st.write("DataFrame shape is",{data.shape})

    st.write("Missing values summary:")
    if st.dataframe(data.isnull().sum()==0):
        st.write(f"The number of missing values are zero")
    else:
        st.write(f"The dataset includes missing values")

    st.markdown("<span style='color:green'>Note: Certain coins will have lesser rows compared to Bitcoin,"
                " Since many coins were recently introduced in the market</span>", unsafe_allow_html=True)

    st.write(f"Summary Statistics of {choice} are:")
    st.dataframe(data.describe())
    st.write("Close Data")
    st.markdown("<span style='color:green'>Note: Hover on the chart and,"
                "click the arrow keys on the top right corner to view the chart for the entire time frame </span>",
                unsafe_allow_html=True)
    st.line_chart(data['close'])

def visualization():
    data_copy = data.copy()
    option = st.radio("Choose a question you want answered:", ['Price over a specified interval',
                                                               '10 positive & negatively correlated coins of selected coin',
                                                               'Moving average of the chosen coin',
                                                               'RSS feed for the chosen coin'])
    #1
    def candlestick():
        figure = go.Figure(data=[go.Candlestick(x=data.index,
                                                open=data['open'],
                                                high=data['high'],
                                                low=data['low'],
                                                close=data['close'])])

        figure.update_layout(
            title=f'Candlestick chart for graphical representation of {choice} over chosen interval of time',
            yaxis_title='Price',
            xaxis_title='Date',
            xaxis_rangeslider_visible=False,
            width=800,
            height=600
        )
        return st.plotly_chart(figure)


    #2
    def correlation_coins(coin):
        #creating an empty DataFrame with date as index
        coin_data = {}

        for symbol in symbols:
            coin_data[symbol] = load_data(symbol, start_date, end_date)

        # combining the close prices of every coin into a single DataFrame
        coin_close_values = pd.DataFrame()
        for symbol, df in coin_data.items():
            coin_close_values[symbol] = df['close']

        corr_matrix = coin_close_values.corr()
        sorted_correlations = corr_matrix[coin].sort_values(ascending=False)
        st.write(f"The top 10 positively correlated coins to {coin} are:")
        st.write(sorted_correlations.head(11))
        st.write(f"The top 10 weakly correlated coins to {coin} are:")
        st.write(sorted_correlations.tail(10))

    #3
    #Moving average of a chosen cryptocurrency
    def moving_average(data, window_size):
        data_copy = data.copy()
        data_copy['moving_avg'] = data_copy['close'].rolling(window=window_size).mean()
        return data_copy



    def news_feed(crypto):
        url = f'https://news.google.com/rss/search?q={crypto}+cryptocurrency&hl=en-US&gl=US&ceid=US:en'
        feed = feedparser.parse(url)
        # returning top-5 stories
        return feed.entries[:5]





    # App display selection
    if option == 'Price over a specified interval':
        candlestick()
    elif option == '10 positive & negatively correlated coins of selected coin':
        coin = st.selectbox("Choose a coin", options=symbols)
        correlation_coins(coin)

    elif option == 'Moving average of the chosen coin':
        window_size = st.slider("Choose the window size for the moving average", min_value=1, max_value=100, value=20,
                                step=1)
        data_with_MA = moving_average(data_copy, window_size)
        st.write(f"Moving average of {choice} with window size {window_size}:")
        st.line_chart(data_with_MA[['close', 'moving_avg']])



    elif option == 'RSS feed for the chosen coin':
        chosen_coin = st.selectbox("Choose a coin", options=symbols)
        top_stories = news_feed(chosen_coin)

        for i, story in enumerate(top_stories):
            st.write(f"{i+1}. {story.title}")
            st.write(f"Link: {story.link}")
            st.write(f"Published on: {story.published}")
            st.write("\n")
        
def display_help():
    st.markdown("<h2 style='color:green;'>How to run the application</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px;color:blue;'>Click on Select a Coin</p>"
                "<p style='font-size:18px;color:blue;'>Either enter the duration of the coin data or enter the time interval you need the data from</p>"
                "<p style='font-size:18px;color:blue;'>Click on SEND if you have selected the dates, this will check if the dates are valid</p>"
                "<p style='font-size:18px;color:blue;'>Click on MAKE A CHOICE, from the option you can check the details about your data, check the important insights in your data and make predictions</p>", 
                unsafe_allow_html=True)


def predict():
    choose_model = st.radio("Choose a prediction model", ['LSTM','SARIMA'])
    num_of_days = st.number_input("Enter how many days to predict",value = 5)
    num_of_days = int(num_of_days)
    if st.button('Predict'):
        
        if choose_model == 'LSTM':
            data = load_data(choice, start_date, end_date)
            mse, rmse,mae,r2,preds_acts, future_prices_df, final_forecast_df = lstm_model(data, num_of_days, choice)
            #st.write(f"Final Loss: {final_loss:.3f}")
            #st.write(f"MSE: {mse:.3f}")
            st.write(f"RMSE: {rmse:.3f}")
            st.write(f"MAE: {mae:.3f}")
            st.write(f"R-SQUARED: {r2:.3f}")
            st.write(future_prices_df)
            
            fig = fig_act_pred(final_forecast_df["Actuals"], final_forecast_df["Predictions"], final_forecast_df.index)

            st.plotly_chart(fig)

        elif choose_model == 'SARIMA':
            data = load_data(choice, start_date, end_date)      
            order = SARIMA_PDQ_DICT[choice]['pdq']
            seasonal_order = SARIMA_PDQ_DICT[choice]['seasonal_pdq']
            sarima_actual, sarima_pred = sarima_model(data, num_of_days, order, seasonal_order)
            #st.write(sarima_pred)
            #st.write(sarima_actual)
            fig = sarima_viz(sarima_actual['close'], sarima_pred['predicted_mean'], sarima_actual['date'], sarima_pred['index'])
            st.plotly_chart(fig)


def fig_act_pred(actual, predicted, date_col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, x= date_col,
                        mode='lines',
                        name='Actual'))
    
    fig.add_trace(go.Scatter(y=predicted, x= date_col,
                        mode='lines',
                        name='Predictions'))
    
    fig.update_layout({"title":{"text":"Actual vs Predicted"}})
    return fig


def sarima_model(data, num_of_days, order, seasonal_order):
    model_data = data['close']

    train_size = int(len(model_data) * 0.80)
    train_row_idx = [i for i in range(train_size)]
    
    model = sm.tsa.statespace.SARIMAX(model_data, order=order, seasonal_order=seasonal_order)
    results = model.fit()
    pred = results.predict(0, len(model_data)+num_of_days-1)
    pred = pd.DataFrame(pred)
    forecast_pred = pred.tail(num_of_days)
    st.write("R2:", r2_score(pred[:-num_of_days], model_data).round(2))
    st.write("MAE:", mean_absolute_error(pred[:-num_of_days], model_data))
    day = 1
    for i in forecast_pred.predicted_mean:
        st.text(f"Day {day}: {i}")
        day +=1

    model_data = model_data.reset_index()
    model_data = model_data.drop(train_row_idx)
    pred = pred.reset_index()
    pred["index"] = pred["index"].apply(lambda x: x.date())
    pred = pred.drop(train_row_idx)

    return model_data, pred

def sarima_viz(actual, predicted, date_col1, date_col2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, x= date_col1,
                        mode='lines',
                        name='Actual'))
    
    fig.add_trace(go.Scatter(y=predicted, x= date_col2,
                        mode='lines',
                        name='Predictions'))
    
    fig.update_layout({"title":{"text":"Actual vs Predicted"}})
    return fig


def lstm_model(data, num_of_days, choice):
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
    #return x_train, y_train

    #lstm model
    model = tf.keras.models.load_model(f"saved_model/{choice}_model.keras")
    model.summary()
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
    preds_acts.index= data.tail(len(preds_acts)).index
    #return rmse, preds_acts

    #future prediction
    last_day = data.index.max()
    print(last_day)
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
    #future_prices_df = pd.DataFrame(future_preds, columns=['Predicted Close Price'], index=future_dates)
    future_prices_df = pd.DataFrame(future_preds, columns=['Predictions'], index=future_dates)
    final_forecast_df = pd.concat([preds_acts, future_prices_df])
    return mse,mae,rmse,r2, preds_acts, future_prices_df, final_forecast_df

# Step 1
def predict_all(num_of_days):
    future_prices = {}
    for symbol in symbols:
        data = load_data(symbol, start_date, end_date)
        mse, rmse, mae, r2, preds_acts, future_prices_df, final_forecast_df = lstm_model(data, num_of_days, symbol)
        future_price = future_prices_df.iloc[-1, 0]  # get the price on the last future day
        future_prices[symbol] = future_price
    return future_prices

# Step 2
def calculate_profit(investment, profit_target, num_of_days, future_prices):
    max_profit = -float('inf')
    best_coin = None
    for coin, future_price in future_prices.items():
        current_price = data.iloc[-1, 0]  # get the current price of the coin
        potential_profit = (future_price - current_price) * investment
        if potential_profit > profit_target and potential_profit > max_profit:
            max_profit = potential_profit
            best_coin = coin
    return best_coin, max_profit



if __name__ == '__main__':
    main()




































