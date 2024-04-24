import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# streamlit run main.py
# Set up title and sidebar
st.title('💀โปรแกรมวิเคราะห์หุ้นสหรัฐ🏴‍☠️')
st.sidebar.header('ช่องใส่ชื่อย่อหุ้น')

# User input for stock symbol and years of prediction
selected_stock = st.sidebar.text_input('ระบุชื่อย่อหุ้น (เช่น, GOOG, AAPL, MSFT, NVDA):')
n_years = st.sidebar.slider('ช่วงเวลาปีที่ต้องการวิเคราะห์:', 1, 4)

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
data_load_state = st.text('กำลังรอข้อมูล...')
data = load_data(selected_stock)
data_load_state.text('กำลังโหลดข้อมูล... เสร็จสิ้น!')

# Display raw data
st.subheader(f'ข้อมูลย้อนหลังของ {selected_stock}')
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Opening Price", line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Closing Price", line=dict(color='firebrick')))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Display raw data plot
st.subheader('ราคาตลาดช่วงเปิด - ปิด🛒📈📉')
plot_raw_data()

# Predict forecast with Prophet
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader('ผลวิเคราะห์การทำนายข้อมูลหุ้น')
st.write(forecast.tail())

# Display forecast plot
st.subheader(f'ผลการวิเคราะห์ในเวลา {n_years} ปี')
fig1 = plot_plotly(m, forecast )
fig1.update_traces(line=dict(color='orange'))  # Forecast line color
st.plotly_chart(fig1)

# Display forecast components
st.subheader('ผลการวิเคราะห์ในช่วงแต่ละเวลา')
fig2 = m.plot_components(forecast)
st.write(fig2)
