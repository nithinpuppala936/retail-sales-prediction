import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib, os
import gdown
#https://drive.google.com/file/d/1fWab1Vjh8V7InNBGArVF8Yfia5R8mGRs/view?usp=sharing
# Google Drive File ID for model
model_id = "1fWab1Vjh8V7InNBGArVF8Yfia5R8mGRs"
MODEL_PATH = "random_forest_model.pkl"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={model_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Debug: check first few bytes of file
with open(MODEL_PATH, "rb") as f:
    first_bytes = f.read(100)
    st.write("First 50 bytes of model file:", first_bytes[:50])  # shows in Streamlit log
    f.seek(0)  # reset file pointer
    url = "https://drive.google.com/uc?id=1fWab1Vjh8V7InNBGArVF8Yfia5R8mGRs"  # replace with your file ID
    output = "random_forest_model.pkl"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)


model_id = "1P-uH27QikOZ9LyW4L9DCgeAHq7q8XlWt"

# Output file names
dataset_path = "dataset.csv"
model_path = "model.pkl"

# Download only if not already present
import os
if not os.path.exists(dataset_path):
    gdown.download(f"https://drive.google.com/uc?id={dataset_id}", dataset_path, quiet=False)

if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={model_id}", model_path, quiet=False)


def add_date_parts(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # ensure datetime
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_week"] = df["date"].dt.dayofweek  # Monday=0, Sunday=6
    return df


# ---------------------------
# Load model & metadata
# ---------------------------
st.set_page_config(page_title="Retail Sales Prediction", page_icon="🛍️", layout="centered")

try:
    model = joblib.load("random_forest_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

try:
    with open("model_meta.json", "r") as f:
        META = json.load(f)
    FEATURES = META["features"]
    CATEGORY_OPTIONS = META["categories"]
except Exception:
    # Fallback if meta is missing
    FEATURES = ['date','store_id','product_id','category','price','promotion','holiday','units_sold']
    CATEGORY_OPTIONS = ['Beverages','Electronics','Home Decor','Clothing','Groceries','Stationery']

# ---------------------------
# Light blue / light white styling
# ---------------------------
st.markdown("""
<style>
  .stApp { background-color: #f0f8ff; }          /* light blue */
  .block-container { background: #ffffff; padding: 24px; border-radius: 16px; 
                     box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
  h1, h2, h3, label { color: #0d1b2a !important; font-weight: 700; }
  .stButton>button { background:#87cefa; color:#000; border:none; border-radius:12px; 
                     padding:10px 20px; font-weight:700; }
  .stButton>button:hover { background:#4682b4; color:#fff; }
    .stSuccess {
        background-color: #d4edda !important;  /* light green */
        color: #155724 !important;  /* dark green text */
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
</style>
""", unsafe_allow_html=True)

st.title("🛍️ Retail Sales Prediction")
st.write("Predict **Total Sales** from store, product, price, promotions and date.")

# ---------------------------
# Inputs (exact raw columns expected by the pipeline)
# ---------------------------
c1, c2 = st.columns(2)

with c1:
    the_date = st.date_input("Date", value=date_cls(2022, 1, 1))
    store_id = st.number_input("Store ID", min_value=1, step=1)
    product_id = st.number_input("Product ID", min_value=1, step=1)
    category = st.selectbox("Category", CATEGORY_OPTIONS)

with c2:
    price = st.number_input("Price", min_value=0.0, step=0.5, format="%.2f")
    promotion = st.selectbox("Promotion (0=No, 1=Yes)", [0, 1])
    holiday = st.selectbox("Holiday (0=No, 1=Yes)", [0, 1])
    units_sold = st.number_input("Units Sold", min_value=0, step=1)

# Build the single-row DataFrame in the **same order** as training
row = {
    'date': str(the_date),  # keep as 'YYYY-MM-DD'; pipeline will create day_of_week & month
    'store_id': int(store_id),
    'product_id': int(product_id),
    'category': category,
    'price': float(price),
    'promotion': int(promotion),
    'holiday': int(holiday),
    'units_sold': int(units_sold),
}
df_in = pd.DataFrame([[row[col] for col in FEATURES]], columns=FEATURES)

# ---------------------------
# Predict
# ---------------------------
if st.button("🔮 Predict Total Sales"):
    try:
        
        prediction = model.predict(df_in)[0]

# Custom styled prediction box
        st.markdown(
    f"""
    <div style="background-color:#d4edda; color:#155724; 
                padding:15px; border-radius:12px; 
                font-weight:bold; font-size:20px; 
                text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.1);">
        💰 Predicted Total Sales: {prediction:.2f}
    </div>
    """,
    unsafe_allow_html=True
)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")


# Load dataset (for visualization only)
@st.cache_data
def load_data():
    return pd.read_csv("retail_sales_50k.csv", parse_dates=["date"])

data = load_data()

st.sidebar.header("🔎 Visualization Options")
viz_option = st.sidebar.selectbox(
    "Choose a Visualization",
    ["Sales Trend Over Time", "Sales by Category", "Sales by Region", "Top 10 Products"]
)

st.subheader("📊 Retail Sales Analysis")

if viz_option == "Sales Trend Over Time":
    fig, ax = plt.subplots(figsize=(10, 5))
    daily_sales = data.groupby("date")["total_sales"].sum()
    daily_sales.plot(ax=ax)
    ax.set_title("Sales Trend Over Time")
    ax.set_ylabel("Total Sales")
    st.pyplot(fig)

elif viz_option == "Sales by Category":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="category", y="total_sales", data=data, estimator=sum, ax=ax)
    ax.set_title("Sales by Category")
    st.pyplot(fig)

elif viz_option == "Sales by Region":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="region", y="total_sales", data=data, estimator=sum, ax=ax)
    ax.set_title("Sales by Region")
    st.pyplot(fig)

elif viz_option == "Top 10 Products":
    fig, ax = plt.subplots(figsize=(10, 5))
    top_products = data.groupby("product")["total_sales"].sum().nlargest(10)
    sns.barplot(x=top_products.index, y=top_products.values, ax=ax)
    ax.set_title("Top 10 Products by Sales")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    st.pyplot(fig)














