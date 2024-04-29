import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# streamlit run main.py
# Set up title and sidebar
st.title('💀AI Stock Prediction🏴‍☠️')
st.sidebar.header('Input Stock Symbol')

# User input for stock symbol and years of prediction
selected_stock = st.sidebar.text_input('(GOOG, AAPL, MSFT, NVDA):')
n_years = st.sidebar.slider('Years of Prediction:', 1, 4)

# Date ranges
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
period = n_years * 365

# Function to load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text('Loading Data...')
data = load_data(selected_stock)
data_load_state.text('Loading Data... Success!')

# Display raw data
st.subheader(f'Display raw data"{selected_stock}"')
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Opening Price", line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Closing Price", line=dict(color='firebrick')))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Display raw data plot
st.subheader('Display raw data plot OPEN - CLOSE Market🛒📈📉')
plot_raw_data()

# Predict forecast with Prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('Display forecast data')
st.write(forecast.tail())

# Display forecast plot
st.subheader(f'Display forecast plot {n_years} years')
fig1 = plot_plotly(m, forecast )
fig1.update_traces(line=dict(color='orange'))  # Forecast line color
st.plotly_chart(fig1)

# Display forecast components
st.subheader('Display forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

# Additional chart - Pie chart for daily price change proportion
st.subheader('Daily Price Change Proportion')
price_increase_days = data[data['Close'] > data['Open']].shape[0]
price_decrease_days = data[data['Close'] < data['Open']].shape[0]
total_days = len(data)
fig_pie = go.Figure(data=[go.Pie(labels=['Price Increase', 'Price Decrease'],
                                  values=[price_increase_days/total_days, price_decrease_days/total_days])])
st.plotly_chart(fig_pie)

# Bar chart for actual vs forecasted closing prices
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=data['Date'], y=data['Close'], name='Actual Closing Price', marker=dict(color='royalblue')))
fig_bar.add_trace(go.Bar(x=forecast['ds'], y=forecast['yhat'], name='Forecasted Closing Price', marker=dict(color='orange')))
fig_bar.update_layout(title='Actual vs Forecasted Closing Prices', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_bar)

# Scatter plot for actual vs forecasted closing prices
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=data['Close'], y=forecast['yhat'], mode='markers', name='Actual vs Forecasted',
                                 marker=dict(color='green', size=8, opacity=0.5)))
fig_scatter.update_layout(title='Actual vs Forecasted Closing Prices', xaxis_title='Actual Closing Price',
                           yaxis_title='Forecasted Closing Price')
st.plotly_chart(fig_scatter)
