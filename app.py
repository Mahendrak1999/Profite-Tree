# pip install streamlit fbprophet yfinance plotly
import pandas as pd
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = {
    'META': 'Meta Platforms Inc.',
    'CRM': 'Salesforce.com Inc.',
    'PYPL': 'PayPal Holdings Inc.',
    'MA': 'Mastercard Incorporated',
    'SHOP': 'Shopify Inc.',
    'V': 'Visa Inc.',
    'UPST': 'Upstart Holdings Inc.',
    'AXP': 'American Express Company',
    'SOFI': 'SoFi Technologies Inc.',
    'WFC': 'Wells Fargo & Company',
    'GS': 'The Goldman Sachs Group Inc.',
    'SONY': 'Sony Group Corporation',
    'ADBE': 'Adobe Inc.',
    'GPRO': 'GoPro Inc.',
    'NVDA': 'NVIDIA Corporation',
    'GOOG': 'Alphabet Inc.',
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GME': 'GameStop Corp.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.'
}

selected_stock = st.sidebar.selectbox('Select dataset for prediction', list(stocks.keys()),
                                      format_func=lambda x: stocks[x])
st.title(" The Stock Information Of" + selected_stock)

if st.sidebar.button('Stock Info Plus'):
    # Clear the page content
    st.empty()

    st.sidebar.markdown("### Stock Info Plus")
    st.sidebar.markdown("Enhancing Your Stock Market Insights")

    # Fetching stock information
    stock_info = yf.Ticker(selected_stock)
    info = stock_info.info

    # Add a heading
    st.markdown("## **Basic Information**")

    # Create 2 columns
    col1, col2 = st.columns(2)

    # Row 1
    col1.dataframe(
        pd.DataFrame({"Issuer Name": [stocks[selected_stock]]}),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame({"Symbol": [selected_stock]}),
        hide_index=True,
        width=500,
    )

    # Row 2
    col1.dataframe(
        pd.DataFrame({"Currency": [info['currency']]}),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame({"Exchange": [info['exchange']]}),
        hide_index=True,
        width=500
    )

    # Add a heading for Market Data
    st.markdown("## **Market Data**")

    # Create 2 columns for Market Data
    col1, col2 = st.columns(2)

    # Fetching stock information
    stock_info = yf.Ticker(selected_stock)
    info = stock_info.info

    # Row 1: Today's stock price
    today_price = info.get('regularMarketPrice', 'NA')
    if today_price == 'NA':
        today_price = info.get('previousClose', 'NA')
    if today_price != 'NA':
        today_price_with_currency = f"{info['currency']} {today_price:.2f}"
    else:
        today_price_with_currency = 'NA'
    col1.dataframe(
        pd.DataFrame({"Today's Stock Price": [today_price_with_currency]}),
        hide_index=True,
        width=500,
    )

    # Row 2: Highest and Lowest stock price
    highest_price = info.get('regularMarketDayHigh', 'NA')
    lowest_price = info.get('regularMarketDayLow', 'NA')

    if highest_price != 'NA':
        highest_price_with_currency = f"{info['currency']} {highest_price:.2f}"
    else:
        highest_price_with_currency = 'NA'

    if lowest_price != 'NA':
        lowest_price_with_currency = f"{info['currency']} {lowest_price:.2f}"
    else:
        lowest_price_with_currency = 'NA'

    col1.dataframe(
        pd.DataFrame({"Highest Stock Price": [highest_price_with_currency]}),
        hide_index=True,
        width=500,
    )
    col2.dataframe(
        pd.DataFrame({"Lowest Stock Price": [lowest_price_with_currency]}),
        hide_index=True,
        width=500
    )

    # Add a heading for Dividends and Yield
    st.markdown("## **Dividends and Yield**")

    # Create 2 columns for Dividends and Yield
    col1, col2 = st.columns(2)

    # Row 1: Dividend Rate and Dividend Yield
    dividend_rate = info.get('dividendRate', 'NA')
    dividend_yield = info.get('dividendYield', 'NA')
    col1.write("Dividend Rate:")
    col1.write(dividend_rate)
    col2.write("Dividend Yield:")
    col2.write(dividend_yield)

    # Row 2: PE Ratio (TTM), Forward PE Ratio, PEG Ratio, Price to Sales Ratio
    pe_ratio_ttm = info.get('trailingPE', 'NA')
    forward_pe_ratio = info.get('forwardPE', 'NA')
    peg_ratio = info.get('pegRatio', 'NA')
    price_to_sales_ratio = info.get('priceToSalesTrailing12Months', 'NA')
    col1.write("PE Ratio (TTM):")
    col1.write(pe_ratio_ttm)
    col2.write("Forward PE Ratio:")
    col2.write(forward_pe_ratio)
    col1.write("PEG Ratio (5 yr expected):")
    col1.write(peg_ratio)
    col2.write("Price to Sales Ratio (TTM):")
    col2.write(price_to_sales_ratio)



    # Row 3: Revenue, Gross Profit, Operating Income
    revenue = info.get('revenue', 'NA')
    gross_profit = info.get('grossProfits', 'NA')
    operating_income = info.get('operatingIncome', 'NA')
    col1.write("Revenue (TTM):")
    col1.write(revenue)
    col2.write("Gross Profit (TTM):")
    col2.write(gross_profit)
    col1.write("Operating Income (TTM):")
    col1.write(operating_income)


n_years = st.sidebar.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)



plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
