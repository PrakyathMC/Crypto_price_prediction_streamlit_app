from tiingo import TiingoClient
import pandas as pd
config = {}

config['session'] = True
config['api_key'] = "d8c623538ff22efb5b7342a198b63b1e430e93f3"

# Initialize
client = TiingoClient(config)

def get_tiingo_data(symbol, startDate, endDate):
    historical_prices = client.get_crypto_price_history(tickers = [symbol], startDate=startDate, endDate=endDate, resampleFreq='1Day')

    df = pd.json_normalize(historical_prices[0]['priceData'])
    df.drop(columns=['volumeNotional','tradesDone'], inplace=True)
    return df

