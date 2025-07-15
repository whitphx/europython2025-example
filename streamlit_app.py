import streamlit as st
import pandas as pd
import altair as alt

from main import process_sales_data


@st.cache_data
def cached_process_sales_data(df):
    return process_sales_data(df)


st.title("Sales Data Processor")

st.write("Upload a CSV file to process:")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
df = cached_process_sales_data(df)

st.write(df)
st.download_button(
    label="Download processed data",
    data=df.to_csv(index=False),
    file_name="processed_sales_data.csv",
    mime="text/csv",
)

### Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total orders", value=len(df))
with col2:
    total_revenue = df["purchase_amount"].sum()
    if total_revenue >= 1_000_000:
        revenue_display = f"${total_revenue / 1_000_000:.1f}M"
    elif total_revenue >= 1_000:
        revenue_display = f"${total_revenue / 1_000:.0f}K"
    else:
        revenue_display = f"${total_revenue:,.0f}"
    st.metric(label="Total revenue", value=revenue_display)
with col3:
    avg_order_value = df["purchase_amount"].mean()
    st.metric(label="Average order value", value=f"${avg_order_value:,.0f}")
with col4:
    positive_sentiment = len(df[df["sentiment"] == "POSITIVE"])
    sentiment_rate = positive_sentiment / len(df) * 100
    st.metric(label="Positive sentiment", value=f"{sentiment_rate:.1f}%")

### Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Customer Segment")
    segment_revenue = (
        df.groupby("customer_segment")["purchase_amount"].sum().reset_index()
    )
    st.bar_chart(segment_revenue.set_index("customer_segment")["purchase_amount"])


with col2:
    st.subheader("Top 10 States by Revenue")
    state_revenue = (
        df.groupby("state")["purchase_amount"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    st.altair_chart(
        alt.Chart(state_revenue.reset_index())
        .mark_bar()
        .encode(x=alt.X("state", sort=None), y=alt.Y("purchase_amount")),
        use_container_width=True,
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Payment Method Distribution")
    payment_counts = df["payment_method"].value_counts()
    st.altair_chart(
        alt.Chart(payment_counts.reset_index())
        .mark_bar()
        .encode(x=alt.X("payment_method", sort=None), y=alt.Y("count")),
        use_container_width=True,
    )

with col2:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    st.altair_chart(
        alt.Chart(sentiment_counts.reset_index())
        .mark_bar()
        .encode(x=alt.X("sentiment", sort=None), y=alt.Y("count")),
        use_container_width=True,
    )
