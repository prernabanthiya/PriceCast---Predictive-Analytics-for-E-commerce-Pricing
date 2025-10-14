import streamlit as st
import time
from PIL import Image
from models import predict_price

# Page config
st.set_page_config(
    page_title="PriceCast - AI Price Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Perplexity-style Professional CSS
st.markdown("""
    <style>
    .stApp {
        background: #0F1419;
        color: #E8E8EA;
    }

    /* Navigation bar */
    .nav-bar {
        background: rgba(21, 27, 34, 0.95);
        backdrop-filter: blur(20px);
        padding: 18px 50px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        margin-bottom: 50px;
    }

    .nav-left {
        display: flex;
        flex-direction: column;
    }

    .nav-logo {
        font-size: 44px;
        font-weight: 900;
        color: #20808D;
        letter-spacing: -1.2px;
        margin-bottom: 3px;
    }

    .nav-tagline {
        font-size: 13px;
        color: #B4B4B8;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    .nav-links {
        display: flex;
        gap: 25px;
        align-items: center;
    }

    .nav-link {
        color: #B4B4B8;
        text-decoration: none;
        font-size: 15px;
        font-weight: 500;
        transition: all 0.2s;
        cursor: pointer;
        padding: 9px 16px;
        border-radius: 7px;
    }

    .nav-link:hover {
        color: #20808D;
        background: rgba(32, 128, 141, 0.1);
    }

    /* Container */
    .main-container {
        max-width: 1250px;
        margin: 0 auto;
        padding: 0 50px;
    }

    /* Typography */
    h1 {
        color: #E8E8EA !important;
        font-size: 44px !important;
        font-weight: 700 !important;
        margin-bottom: 18px !important;
        letter-spacing: -1px;
    }

    h3 {
        color: #E8E8EA !important;
        font-size: 21px !important;
        font-weight: 600 !important;
        margin-bottom: 22px !important;
    }

    /* Input styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background: #191F28 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 9px !important;
        color: #E8E8EA !important;
        padding: 13px 17px !important;
        font-size: 15px !important;
        transition: all 0.2s !important;
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #20808D !important;
        box-shadow: 0 0 0 3px rgba(32, 128, 141, 0.12) !important;
        background: #1F2937 !important;
    }

    .stTextInput>div>div>input::placeholder,
    .stTextArea>div>div>textarea::placeholder {
        color: #6B7280 !important;
    }

    .stTextInput>label, .stTextArea>label, .stSelectbox>label {
        color: #B4B4B8 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        margin-bottom: 9px !important;
    }

    /* Button */
    .stButton>button {
        background: #20808D !important;
        color: white !important;
        border: none !important;
        border-radius: 9px !important;
        padding: 15px 30px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
        width: 100%;
    }

    .stButton>button:hover {
        background: #1a6b76 !important;
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(32, 128, 141, 0.25) !important;
    }

    /* File uploader */
    .stFileUploader {
        background: #191F28;
        border: 2px dashed rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 35px;
        text-align: center;
        transition: all 0.3s;
    }

    .stFileUploader:hover {
        border-color: rgba(32, 128, 141, 0.4);
        background: #1F2937;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #191F28 0%, #1F2937 100%);
        border: 1px solid rgba(32, 128, 141, 0.35);
        border-radius: 16px;
        padding: 42px;
        text-align: center;
        margin: 32px 0;
    }

    .price-value {
        font-size: 58px;
        font-weight: 800;
        color: #20808D;
        margin: 22px 0;
        letter-spacing: -1.5px;
    }

    .price-range {
        color: #B4B4B8;
        font-size: 16px;
    }

    /* Metric boxes */
    .metric-box {
        background: linear-gradient(135deg, #20808D 0%, #1a6b76 100%);
        border-radius: 12px;
        padding: 26px 20px;
        text-align: center;
        transition: all 0.3s;
    }

    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 28px rgba(32, 128, 141, 0.2);
    }

    .metric-title {
        color: rgba(255, 255, 255, 0.8);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin-bottom: 11px;
    }

    .metric-value {
        color: white;
        font-size: 29px;
        font-weight: 700;
        margin: 9px 0;
    }

    .metric-weight {
        color: rgba(255, 255, 255, 0.65);
        font-size: 13px;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #20808D, #1a6b76);
    }

    /* Empty state */
    .empty-state {
        background: #191F28;
        border: 2px dashed rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 65px 32px;
        text-align: center;
        color: #B4B4B8;
    }

    .stImage {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        overflow: hidden;
    }

    hr {
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 32px 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Navigation Bar
st.markdown("""
    <div class='nav-bar'>
        <div class='nav-left'>
            <div class='nav-logo'>PriceCast</div>
            <div class='nav-tagline'>Predictive Analytics for E-commerce Pricing</div>
        </div>
        <div class='nav-links'>
            <a class='nav-link' href='/' target='_self'>Predictor</a>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Main content
st.markdown("<h1>Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("<h3>Product Details</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload product image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear product image"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    st.markdown("<div style='margin-top: 26px;'></div>", unsafe_allow_html=True)

    title = st.text_input(
        "Product Title",
        placeholder="Enter product name"
    )

    col_cat, col_desc = st.columns([1, 2])

    with col_cat:
        category = st.selectbox(
            "Category",
            ["Electronics", "Clothing", "Food & Grocery", "Home & Garden",
             "Sports", "Books", "Beauty", "Toys", "Other"]
        )

    description = st.text_area(
        "Description (Optional)",
        height=125,
        placeholder="Add product description for better accuracy"
    )

    st.markdown("<div style='margin-top: 26px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Generate Prediction")

with col2:
    st.markdown("<h3>Results</h3>", unsafe_allow_html=True)

    if predict_btn:
        if not uploaded_file or not title:
            st.error("Please upload image and enter product title")
        else:
            with st.spinner("Processing..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)

                result = predict_price(image, title, description, category)
                progress.empty()

            st.markdown(f"""
                <div class='result-card'>
                    <p style='color: #B4B4B8; font-size: 14px; margin: 0;'>Predicted Price</p>
                    <div class='price-value'>${result['price']}</div>
                    <p class='price-range'>Range: ${result['range_low']} - ${result['range_high']}</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(
                "<p style='color: #E8E8EA; font-size: 16px; font-weight: 600; margin: 32px 0 18px 0;'>Model Contributions</p>",
                unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)

            with m1:
                st.markdown(f"""
                    <div class='metric-box'>
                        <p class='metric-title'>MLP Network</p>
                        <p class='metric-value'>${result['mlp']}</p>
                        <p class='metric-weight'>40% weight</p>
                    </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                    <div class='metric-box'>
                        <p class='metric-title'>Ridge Regression</p>
                        <p class='metric-value'>${result['ridge']}</p>
                        <p class='metric-weight'>35% weight</p>
                    </div>
                """, unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                    <div class='metric-box'>
                        <p class='metric-title'>LightGBM</p>
                        <p class='metric-value'>${result['lgb']}</p>
                        <p class='metric-weight'>25% weight</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            st.markdown("<p style='color: #B4B4B8; font-size: 14px; margin-bottom: 13px;'>Prediction Confidence</p>",
                        unsafe_allow_html=True)
            st.progress(result['confidence'])
            st.markdown(
                f"<p style='text-align: center; color: #20808D; font-size: 19px; font-weight: 600; margin-top: 9px;'>{int(result['confidence'] * 100)}%</p>",
                unsafe_allow_html=True)

    else:
        st.markdown("""
            <div class='empty-state'>
                <p style='font-size: 50px; margin: 0;'>📊</p>
                <p style='font-size: 17px; margin-top: 17px; font-weight: 500;'>No prediction yet</p>
                <p style='font-size: 14px; margin-top: 9px; opacity: 0.7;'>Upload image and fill details to get started</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
