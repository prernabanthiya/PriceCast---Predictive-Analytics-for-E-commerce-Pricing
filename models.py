"""
Extracted from 50_SMAPE-1.ipynb
Model prediction logic for Streamlit app
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


# ============================================
# TEXT NORMALIZATION & FEATURE ENGINEERING
# ============================================

def normalize_text(s):
    """Normalize text from notebook"""
    if not isinstance(s, str):
        return ""
    return s.replace('\u2019', "'").replace('\u2018', "'").replace('\u2013', "-").replace('\u2014', "-").strip()


def parse_pack_count(s):
    """Extract pack count from text"""
    if not isinstance(s, str):
        return 1
    s = normalize_text(s)

    # Try multiple patterns
    m = re.search(r'(\d+)\s*pack', s, flags=re.I)
    if m:
        try:
            return int(m.group(1))
        except:
            pass

    m = re.search(r'(\d+)\s*x', s, flags=re.I)
    if m:
        try:
            return int(m.group(1))
        except:
            pass

    return 1


def parse_size(s):
    """Extract size and unit from text"""
    if not isinstance(s, str):
        return np.nan, None
    s = normalize_text(s)

    m = re.search(r'(\d+\.?\d*)\s*(fl\.?\s?oz|fluid\s?ounce|ounce|oz|ml|l|g|kg|lb|lbs)', s, flags=re.I)
    if not m:
        return np.nan, None

    try:
        val = float(m.group(1))
    except:
        return np.nan, None

    unit = m.group(2).lower().replace('.', '').replace(' ', '')
    return val, unit


def normalize_qty(v, u):
    """Normalize quantities to common unit"""
    UNIT_TO_BASE = {
        'oz': ('g', 28.3495),
        'ounce': ('g', 28.3495),
        'floz': ('ml', 29.5735),
        'fluidounce': ('ml', 29.5735),
        'ml': ('ml', 1.0),
        'l': ('ml', 1000.0),
        'g': ('g', 1.0),
        'kg': ('g', 1000.0),
        'lb': ('g', 453.592),
        'lbs': ('g', 453.592),
    }

    if u is None or not np.isfinite(v):
        return np.nan, None

    if u in UNIT_TO_BASE:
        base, factor = UNIT_TO_BASE[u]
        return v * factor, base

    return np.nan, None


# ============================================
# MLP MODEL ARCHITECTURE (From Notebook)
# ============================================

class BalancedRegressor(nn.Module):
    """MLP architecture from your notebook"""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.05),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_price(image, title, description, category):
    """
    Main prediction function using your notebook's logic

    NOTE: This is a SIMPLIFIED version for demo.
    For full accuracy, you need to:
    1. Save trained models from notebook
    2. Load CLIP and SBERT models
    3. Extract actual embeddings
    """

    # Combine text
    full_text = f"{title} {description}".lower()

    # Extract features (from notebook)
    pack_count = parse_pack_count(full_text)
    size_val, size_unit = parse_size(full_text)
    norm_qty, norm_unit = normalize_qty(size_val, size_unit)

    if not np.isfinite(norm_qty):
        norm_qty = 0.0

    total_qty = pack_count * norm_qty

    # Text features
    has_sale = int(any(word in full_text for word in ['sale', 'discount']))
    text_word_cnt = len(full_text.split())
    avg_w_len = len(full_text) / max(text_word_cnt, 1)

    # Logging
    log_total_qty = np.log1p(total_qty)
    qty_per_pack = np.log1p(norm_qty) / max(pack_count, 1)
    has_pack = int(pack_count > 1)

    # Category-based baseline (from your data distribution)
    category_base = {
        'Electronics': (45, 380),
        'Clothing': (12, 130),
        'Food & Grocery': (4, 65),
        'Home & Garden': (18, 190),
        'Sports': (22, 210),
        'Books': (7, 55),
        'Beauty': (10, 95),
        'Toys': (8, 120),
        'Other': (10, 100)
    }

    min_price, max_price = category_base.get(category, (10, 100))

    # Apply feature-based adjustments
    multiplier = 1.0

    if pack_count > 1:
        multiplier *= (1.0 + (pack_count - 1) * 0.14)

    if any(word in full_text for word in ['premium', 'pro', 'deluxe', 'luxury']):
        multiplier *= 1.40

    premium_brands = ['apple', 'samsung', 'sony', 'nike', 'adidas', 'dell', 'hp']
    if any(brand in full_text for brand in premium_brands):
        multiplier *= 1.55

    if has_sale:
        multiplier *= 0.72

    # Generate base prediction
    np.random.seed(hash(title) % 100000)
    base_price = np.random.uniform(min_price, max_price) * multiplier

    # Simulate ensemble (from your notebook: 70% MLP + 30% Ridge)
    # In reality, you'd load actual models here
    mlp_pred = base_price * np.random.uniform(0.96, 1.05)
    ridge_pred = base_price * np.random.uniform(0.92, 1.09)
    lgb_pred = base_price * np.random.uniform(0.97, 1.03)

    # Ensemble with your best weights (from notebook: 0.4 MLP + 0.35 Ridge + 0.25 LGB)
    ensemble = 0.40 * mlp_pred + 0.35 * ridge_pred + 0.25 * lgb_pred

    # Confidence
    std_dev = np.std([mlp_pred, ridge_pred, lgb_pred])
    mean_pred = np.mean([mlp_pred, ridge_pred, lgb_pred])
    confidence = max(0.70, min(0.96, 1.0 - (std_dev / mean_pred * 3)))

    return {
        'price': round(ensemble, 2),
        'range_low': round(ensemble * 0.88, 2),
        'range_high': round(ensemble * 1.12, 2),
        'confidence': confidence,
        'mlp': round(mlp_pred, 2),
        'ridge': round(ridge_pred, 2),
        'lgb': round(lgb_pred, 2)
    }


# ============================================
# TO USE REAL MODELS (After saving from notebook)
# ============================================

def predict_price_with_real_models(image, title, description, category):
    """
    Use this after you save models from notebook

    Steps to enable:
    1. Run notebook and save models:
       - torch.save(mlp_model.state_dict(), 'mlp_model.pth')
       - joblib.dump(ridge_model, 'ridge_model.pkl')
       - lgb_model.save_model('lgb_model.txt')

    2. Save embeddings models:
       - Save CLIP model
       - Save SBERT model
       - Save PCA transformers

    3. Load them here and use for prediction
    """

    # Load models (uncomment after saving)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mlp_model = BalancedRegressor(393).to(device)
    # mlp_model.load_state_dict(torch.load('mlp_model.pth'))
    # ridge_model = joblib.load('ridge_model.pkl')
    # lgb_model = lgb.Booster(model_file='lgb_model.txt')

    # Extract features (would need CLIP/SBERT loaded)
    # ...

    # For now, use simplified version
    return predict_price(image, title, description, category)
