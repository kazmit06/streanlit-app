import streamlit as st
import plotly.express as px

st.title("設定ページ")
st.write("アプリの設定を行います。")

st.write('Plotlyでデータ出力')
data_canada = px.data.gapminder().query("country=='Canada'")
fig = px.bar(data_canada,x='year',y='pop')
st.plotly_chart(fig, use_container_width=True)