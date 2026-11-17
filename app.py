from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np


# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Online Retail Analytics",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------
# 2. LOAD DATA
# --------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("online_retail_cleaned.csv")
    rfm_data = pd.read_csv("customer_rfm_analysis.csv")

    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )

    data["Sales"] = (
        pd.to_numeric(data["Quantity"], errors="coerce") *
        pd.to_numeric(data["UnitPrice"], errors="coerce")
    )

    data["CustomerID"] = data["CustomerID"].astype("string")

    return data, rfm_data

df, rfm = load_data()

# --------------------------------
# 3. APPLICATION TITLE
# --------------------------------

st.title("🛍️ Online Retail Sales Analytics")
st.write(
    "An interactive Business Intelligence Dashboard "
    "for analyzing retail sales and customer behaviour."
)

st.divider()

# --------------------------------
# 4. SIDEBAR FILTER
# --------------------------------

st.sidebar.header("Dashboard Filters")

countries = sorted(df["Country"].dropna().unique())

selected_country = st.sidebar.selectbox(
    "Select Country",
    ["All Countries"] + countries
)

if selected_country == "All Countries":
    filtered = df.copy()
else:
    filtered = df[df["Country"] == selected_country].copy()

# --------------------------------
# 5. KEY PERFORMANCE INDICATORS
# --------------------------------

revenue = filtered["Sales"].sum()

orders = filtered["InvoiceNo"].nunique()

customers = filtered["CustomerID"].nunique()

products = filtered["StockCode"].nunique()

average_order = revenue / orders if orders > 0 else 0

st.subheader("Business Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"£{revenue:,.2f}")
c2.metric("Total Orders", f"{orders:,}")
c3.metric("Total Customers", f"{customers:,}")
c4.metric("Average Order Value", f"£{average_order:,.2f}")

st.divider()

# --------------------------------
# 6. MONTHLY REVENUE ANALYSIS
# --------------------------------

st.subheader("Monthly Revenue Analysis")

monthly = (
    filtered.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

monthly = monthly.set_index("Month")

st.line_chart(monthly)

# --------------------------------
# 7. TOP SELLING PRODUCTS
# --------------------------------

st.subheader("Top 10 Products by Revenue")

top_products = (
    filtered.groupby("Description")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_products = top_products.set_index("Description")

st.bar_chart(top_products)

# --------------------------------
# 8. COUNTRY-WISE SALES
# --------------------------------

st.subheader("Country-wise Revenue")

country_sales = (
    df.groupby("Country")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

country_sales = country_sales.set_index("Country")

st.bar_chart(country_sales)

# --------------------------------
# 9. CUSTOMER SEGMENTATION
# --------------------------------

st.subheader("Customer Segmentation - RFM Analysis")

if "Segment" in rfm.columns:

    segment_count = (
        rfm["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_count.columns = [
        "Segment", "Customers"
    ]

    st.bar_chart(
        segment_count.set_index("Segment")
    )

    st.write("Customer Segmentation Summary")

    st.dataframe(
        segment_count,
        use_container_width=True
    )

else:
    st.warning("Segment column not found.")

# --------------------------------
# 11. DOWNLOAD REPORT
# --------------------------------

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Sales Data",
    data=csv_data,
    file_name="retail_sales_report.csv",
    mime="text/csv"
)

# --------------------------------
# 12. FOOTER
# --------------------------------

st.divider()

st.caption(
    "Developed as a Business Analytics Group Project "
    "| Python | Streamlit | Plotly"
)
