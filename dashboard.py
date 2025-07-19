import streamlit as st
import pandas as pd
from textblob import TextBlob
import openai

st.set_page_config(page_title="SME Insight Dashboard", layout="wide")

st.title("📊 AI-Powered Business Insight Dashboard")

uploaded_file = st.file_uploader("Upload your business data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Raw Data Preview")
    st.write(df.head())

    if "Sales" in df.columns and "Expense" in df.columns:
        df["Profit"] = df["Sales"] - df["Expense"]
        st.subheader("📈 Sales vs Expense")
        st.line_chart(df[["Sales", "Expense", "Profit"]])

    if "Feedback" in df.columns:
        st.subheader("🧠 Customer Feedback Sentiment")
        df["Polarity"] = df["Feedback"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        st.bar_chart(df["Polarity"])
        st.write("Average Sentiment Score:", df["Polarity"].mean())
