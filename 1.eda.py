# pages/eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="EDA", layout="wide")

st.title(" Exploratory Data Analysis (EDA)")
st.markdown("A comprehensive look into the equipment anomaly data.")

# Load dataset
DATA_PATH = os.path.join("data", "equipment_anomaly_data.csv")
df = pd.read_csv(DATA_PATH)

# Sidebar filtering
st.sidebar.header(" Filter Options")

# Get column options
date_col = 'timestamp' if 'timestamp' in df.columns else None
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Filter by category
if cat_cols:
    for col in cat_cols:
        selected = st.sidebar.multiselect(f"Filter by {col}", options=df[col].unique())
        if selected:
            df = df[df[col].isin(selected)]

# Filter by date
if date_col:
    df[date_col] = pd.to_datetime(df[date_col])
    date_range = st.sidebar.date_input("Filter by Date Range", [])
    if len(date_range) == 2:
        df = df[(df[date_col] >= pd.to_datetime(date_range[0])) & (df[date_col] <= pd.to_datetime(date_range[1]))]

# Show data
st.subheader(" Filtered Data Preview")
st.dataframe(df, use_container_width=True)

# Time-Series Plot
if date_col and num_cols:
    st.subheader(" Time-Series Trend")
    ts_col = st.selectbox("Select a numeric column to view over time:", num_cols)
    fig, ax = plt.subplots()
    df_sorted = df.sort_values(by=date_col)
    ax.plot(df_sorted[date_col], df_sorted[ts_col], marker='o', linestyle='-', alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel(ts_col)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Correlation Heatmap
st.subheader(" Correlation Heatmap (Numeric Features)")
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Line Plot (custom)
st.subheader(" Line Plot Comparison")
line_x = st.selectbox("X-axis:", options=num_cols)
line_y = st.selectbox("Y-axis:", options=num_cols, index=1 if len(num_cols) > 1 else 0)
fig2, ax2 = plt.subplots()
ax2.plot(df[line_x], df[line_y], color='green', marker='x')
ax2.set_title(f"{line_y} vs {line_x}")
st.pyplot(fig2)
