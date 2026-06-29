import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("E-COMMERCE SALES ANALYSIS PROJECT")
print("=" * 60)

# Load Datasets

customers = pd.read_csv("data/olist_customers_dataset.csv")

orders = pd.read_csv("data/olist_orders_dataset.csv")

items = pd.read_csv("data/olist_order_items_dataset.csv")

payments = pd.read_csv("data/olist_order_payments_dataset.csv")

reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")

products = pd.read_csv("data/olist_products_dataset.csv")

sellers = pd.read_csv("data/olist_sellers_dataset.csv")

geolocation = pd.read_csv("data/olist_geolocation_dataset.csv")

translation = pd.read_csv(
    "data/product_category_name_translation.csv"
)

print("\nDatasets Loaded Successfully!\n")

print("Customers Dataset :", customers.shape)
print("Orders Dataset    :", orders.shape)
print("Items Dataset     :", items.shape)
print("Payments Dataset  :", payments.shape)
print("Reviews Dataset   :", reviews.shape)
print("Products Dataset  :", products.shape)
print("Sellers Dataset   :", sellers.shape)
print("Geolocation Data  :", geolocation.shape)
print("Translation Data  :", translation.shape)
# =====================================
# DATA CLEANING
# =====================================

print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

datasets = {
    "Customers": customers,
    "Orders": orders,
    "Items": items,
    "Payments": payments,
    "Reviews": reviews,
    "Products": products,
    "Sellers": sellers
}

for name, df in datasets.items():

    print(f"\n{name} Dataset")

    print("Shape:", df.shape)

    print("Missing Values:")
    print(df.isnull().sum().sum())

    print("Duplicate Rows:")
    print(df.duplicated().sum())
# =====================================
# DATE CONVERSION
# =====================================

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

orders["order_approved_at"] = pd.to_datetime(
    orders["order_approved_at"]
)

orders["order_delivered_customer_date"] = pd.to_datetime(
    orders["order_delivered_customer_date"]
)

print("\nDate Conversion Completed Successfully!")

print("\nData Types:")
print(orders[[
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_customer_date"
]].dtypes)
# =====================================
# PRODUCT CATEGORY TRANSLATION
# =====================================

products = products.merge(
    translation,
    on="product_category_name",
    how="left"
)

print("\nProduct Categories Translated Successfully!")

print("\nProducts Dataset Shape:")
print(products.shape)

print("\nSample Categories:")
print(
    products[
        [
            "product_category_name",
            "product_category_name_english"
        ]
    ].head()
)
# =====================================
# MASTER DATASET CREATION
# =====================================

sales = items.merge(
    orders,
    on="order_id"
)

sales = sales.merge(
    customers,
    on="customer_id"
)

sales = sales.merge(
    products,
    on="product_id"
)

sales = sales.merge(
    reviews[["order_id", "review_score"]],
    on="order_id",
    how="left"
)

print("\nMaster Dataset Created Successfully!")

print("\nMaster Dataset Shape:")
print(sales.shape)

print("\nMaster Dataset Columns:")
print(sales.columns.tolist())

# =====================================
# FEATURE ENGINEERING
# =====================================

sales["revenue"] = (
    sales["price"] +
    sales["freight_value"]
)

sales["month"] = (
    sales["order_purchase_timestamp"]
    .dt.month_name()
)

sales["year"] = (
    sales["order_purchase_timestamp"]
    .dt.year
)

print("\nFeature Engineering Completed!")

print("\nRevenue Statistics:")
print(sales["revenue"].describe())

print("\nYears Available:")
print(sales["year"].unique())
# =====================================
# QUESTION 1
# TOP REVENUE CATEGORY
# =====================================

top_category = (
    sales.groupby(
        "product_category_name_english"
    )["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 Revenue Categories:\n")

print(
    top_category.head(10)
)
# =====================================
# VISUALIZATION 1
# =====================================

plt.figure(figsize=(12,6))

top_category.head(10).plot(
    kind="bar"
)

plt.title(
    "Top 10 Revenue Generating Categories"
)

plt.xlabel("Product Category")

plt.ylabel("Revenue")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
# =====================================
# QUESTION 2
# MONTHLY SALES TREND
# =====================================

monthly_sales = (
    sales.groupby("month")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nMonthly Revenue:\n")
print(monthly_sales)
# =====================================
# VISUALIZATION 2
# MONTHLY SALES TREND
# =====================================

monthly_trend = (
    sales.groupby(
        sales["order_purchase_timestamp"]
        .dt.to_period("M")
    )["revenue"]
    .sum()
)

plt.figure(figsize=(12,6))

monthly_trend.plot(
    marker="o"
)

plt.title(
    "Monthly Revenue Trend"
)

plt.xlabel("Month")

plt.ylabel("Revenue")

plt.grid(True)

plt.tight_layout()

plt.show()
# =====================================
# QUESTION 3
# BEST PERFORMING STATE
# =====================================

state_sales = (
    sales.groupby("customer_state")
    ["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTop 10 States by Revenue:\n")

print(
    state_sales.head(10)
)
# =====================================
# VISUALIZATION 3
# TOP STATES BY REVENUE
# =====================================

plt.figure(figsize=(12,6))

state_sales.head(10).plot(
    kind="bar"
)

plt.title(
    "Top 10 States by Revenue"
)

plt.xlabel("State")

plt.ylabel("Revenue")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()
# =====================================
# QUESTION 4
# AVERAGE ORDER VALUE
# =====================================

order_value = (
    sales.groupby("order_id")["revenue"]
    .sum()
)

average_order_value = round(
    order_value.mean(), 2
)

print("\nAverage Order Value:")

print(f"₹ {average_order_value}")

print("\nOrder Value Statistics:")

print(order_value.describe())
# =====================================
# VISUALIZATION 4
# ORDER VALUE DISTRIBUTION
# =====================================

plt.figure(figsize=(10,6))

plt.hist(
    order_value,
    bins=30
)

plt.title(
    "Order Value Distribution"
)

plt.xlabel(
    "Order Value"
)

plt.ylabel(
    "Number of Orders"
)

plt.tight_layout()

plt.savefig(
    "dashboard/order_value_distribution.png"
)

plt.show()
# =====================================
# QUESTION 5
# CUSTOMER REVIEW SCORE DISTRIBUTION
# =====================================

review_distribution = (
    sales["review_score"]
    .value_counts()
    .sort_index()
)

print("\nCustomer Review Score Distribution:\n")

print(review_distribution)

average_review_score = round(
    sales["review_score"].mean(), 2
)

print(f"\nAverage Review Score : {average_review_score}")
# =====================================
# VISUALIZATION 5
# CUSTOMER REVIEW SCORE PIE CHART
# =====================================

plt.figure(figsize=(8,8))

plt.pie(
    review_distribution,
    labels=review_distribution.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Customer Review Score Distribution"
)

plt.tight_layout()

plt.savefig(
    "dashboard/review_score_distribution.png"
)

plt.show()
# =====================================
# VISUALIZATION 6
# HEATMAP - CATEGORY VS MONTH
# =====================================

heatmap_data = pd.pivot_table(
    sales,
    values="revenue",
    index="product_category_name_english",
    columns="month",
    aggfunc="sum",
    fill_value=0
)

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

heatmap_data = heatmap_data.reindex(
    columns=month_order
)

plt.figure(figsize=(14,10))

sns.heatmap(
    heatmap_data,
    cmap="YlGnBu"
)

plt.title(
    "Revenue by Product Category and Month"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Product Category"
)

plt.tight_layout()

plt.savefig(
    "dashboard/category_month_heatmap.png"
)

plt.show()
# =====================================
# KPI DASHBOARD
# =====================================

total_revenue = round(
    sales["revenue"].sum(),
    2
)

total_orders = sales["order_id"].nunique()

average_review = round(
    sales["review_score"].mean(),
    2
)

print("\n" + "="*60)
print("KPI DASHBOARD")
print("="*60)

print(f"Total Revenue      : ₹ {total_revenue:,.2f}")
print(f"Total Orders       : {total_orders}")
print(f"Average Review     : {average_review}")

# =====================================
# KPI SUMMARY
# =====================================

kpi_names = [
    "Revenue",
    "Orders",
    "Review Score"
]

kpi_values = [
    total_revenue,
    total_orders,
    average_review
]

plt.figure(figsize=(8,5))

plt.bar(
    kpi_names,
    kpi_values
)

plt.title(
    "Business KPI Summary"
)

plt.xlabel("KPI")

plt.ylabel("Value")

plt.tight_layout()

plt.show()
