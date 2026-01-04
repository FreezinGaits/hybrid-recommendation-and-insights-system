import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
EMPTY_TEMPORAL_INSIGHTS = {
    "purchase_frequency": 0,
    "avg_order_value": 0.0,
    "recent_co_purchases": [],
    "analysis_window_days": 0,
    "recent_trends": [],
    "monthly_trends": None,
    "recent_user_activity": None,
}


# Custom Streamlit theme with dark mode support
st.set_page_config(
    page_title="Product Bundle Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with dark/light mode support
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Main app styling - Dynamic for dark/light mode */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        min-height: 100vh;
        padding: 20px;
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    }
    
    /* Main container with glass morphism effect */
    .main-container {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 40px;
        margin: 20px auto;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        color: #f1f5f9;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 600px;
        width: 95%;
        max-width: 1400px;
    }
    
    /* Text colors for dark mode */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #f1f5f9 !important;
    }
    
    /* Headers with gradient text */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        background: linear-gradient(45deg, #60a5fa, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Cards with dark mode support */
    .card {
        background: rgba(30, 41, 59, 0.7) !important;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #f1f5f9;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
        background: linear-gradient(45deg, #2563eb, #7c3aed);
    }
    
    /* Sidebar styling */
    .css-1v3fvcr {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f1f5f9;
    }
    
    /* Input widgets */
    .stSelectbox, .stSlider, .stCheckbox, .stNumberInput {
        color: #f1f5f9 !important;
        background: rgba(30, 41, 59, 0.7) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: rgba(30, 41, 59, 0.7) !important;
        color: #f1f5f9 !important;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white !important;
        border-radius: 12px;
        padding: 15px;
    }
    
    .stMetric > div > div {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 41, 59, 0.7);
        color: #f1f5f9;
        border-radius: 0 0 10px 10px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        color: #f1f5f9;
        border-radius: 15px;
        font-size: 14px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
        backdrop-filter: blur(10px);
    }
    
    .badge-apriori { 
        background: linear-gradient(45deg, rgba(255, 154, 158, 0.2), rgba(250, 208, 196, 0.2));
        color: #f1f5f9;
        border: 1px solid rgba(255, 154, 158, 0.3);
    }
    .badge-collab { 
        background: linear-gradient(45deg, rgba(161, 196, 253, 0.2), rgba(194, 233, 251, 0.2));
        color: #f1f5f9;
        border: 1px solid rgba(161, 196, 253, 0.3);
    }
    .badge-content { 
        background: linear-gradient(45deg, rgba(255, 236, 210, 0.2), rgba(252, 182, 159, 0.2));
        color: #f1f5f9;
        border: 1px solid rgba(255, 236, 210, 0.3);
    }
    .badge-graph { 
        background: linear-gradient(45deg, rgba(212, 252, 121, 0.2), rgba(150, 230, 161, 0.2));
        color: #f1f5f9;
        border: 1px solid rgba(212, 252, 121, 0.3);
    }
    .badge-ensemble { 
        background: linear-gradient(45deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2));
        color: #f1f5f9;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Status indicators */
    .status-box {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Fix for empty white boxes */
    .st-emotion-cache-1y4p8pa {
        background: transparent !important;
    }
    
    /* Chart containers */
    .js-plotly-plot, .plot-container {
        background: rgba(30, 41, 59, 0.7) !important;
        border-radius: 12px;
        padding: 15px;
    }
    
    /* Table styling */
    .dataframe {
        background: rgba(30, 41, 59, 0.7) !important;
        color: #f1f5f9 !important;
    }
    
    .dataframe th {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #f1f5f9 !important;
    }
    
    .dataframe td {
        color: #f1f5f9 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Fix for Streamlit's default containers */
    .block-container {
        padding: 2rem 1rem !important;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #3b82f6 transparent transparent transparent !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 10px; background: linear-gradient(45deg, #60a5fa, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎁 Smart Product Bundle Recommender</h1>
        <p style="font-size: 1.2rem; color: #94a3b8; margin-bottom: 30px;">
            AI-powered recommendations combining association rules, collaborative filtering, content-based matching, and graph analytics
        </p>
    </div>
""", unsafe_allow_html=True)

# Loading functions
@st.cache_resource
def load_artifacts():
    """Load all trained models and data artifacts"""
    try:
        artifacts = joblib.load("recommender_artifacts.joblib")
        return (
            artifacts.get('products_df'),
            artifacts.get('label_encoders', {}),
            artifacts.get('bundle_rules'),
            artifacts.get('item_similarity_df'),
            artifacts.get('content_similarity_df'),
            artifacts.get('G'),
            artifacts.get('user_item_matrix'),
            artifacts.get('tfidf_vectorizer', None)
        )
    except Exception as e:
        st.error(f"⚠️ Error loading artifacts: {str(e)[:100]}")
        return None, None, None, None, None, None, None, None

@st.cache_data
def load_raw_data():
    """Load raw e-commerce data with proper validation"""
    try:
        data = pd.read_csv("ECommerceOrderBundles.csv")
        
        # Ensure required columns exist
        required_cols = ['order_date', 'product_id', 'user_id', 'item_total', 'product_name']
        for col in required_cols:
            if col not in data.columns:
                st.error(f"🚨 Missing required column: {col}")
                return None
        
        # Convert date with error handling
        data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
        
        # Fill missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)
        
        # Remove invalid dates
        invalid_dates = data['order_date'].isna().sum()
        if invalid_dates > 0:
            st.info(f"Note: {invalid_dates} records have invalid dates")
            data = data.dropna(subset=['order_date'])
        
        return data
    except Exception as e:
        st.error(f"🚨 Error loading data: {str(e)[:100]}")
        return None

# Load data
raw_df = load_raw_data()
products_df, label_encoders, bundle_rules, item_similarity_df, content_similarity_df, G, user_item_matrix, tfidf_vectorizer = load_artifacts()

# st.write("DEBUG data columns:", raw_df.columns.tolist())

@st.cache_data
def build_raw_product_lookup(df):
    return (
        df[['product_name', 'product_id']]
        .drop_duplicates()
        .set_index('product_name')['product_id']
        .to_dict()
    )

raw_product_lookup = build_raw_product_lookup(raw_df)



def get_association_bundles(product_id, top_n=5):
    """Get bundle recommendations using association rules - IMPROVED"""
    try:
        if bundle_rules is None or len(bundle_rules) == 0:
            return []
        
        # Create a dictionary to track scores
        recommendation_scores = {}
        
        # Process both antecedents and consequents
        for idx, row in bundle_rules.iterrows():
            # Extract products from frozensets
            antecedents = list(row['antecedents']) if hasattr(row['antecedents'], '__iter__') else []
            consequents = list(row['consequents']) if hasattr(row['consequents'], '__iter__') else []
            
            all_products = antecedents + consequents
            
            # If our product is in the rule, score all other products
            if product_id in all_products:
                score = row.get('confidence', 0.1) * row.get('lift', 1.0)
                
                for p in all_products:
                    if p != product_id:
                        if p not in recommendation_scores:
                            recommendation_scores[p] = 0
                        recommendation_scores[p] += score
        
        # Sort by score and return top N
        sorted_recommendations = sorted(recommendation_scores.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        # Return product IDs
        return [rec[0] for rec in sorted_recommendations[:top_n]]
    
    except Exception as e:
        print(f"Association rule error: {e}")
        return []

def get_collaborative_bundles(product_id, top_n=5):
    """Get recommendations using collaborative filtering"""
    try:
        if item_similarity_df is None or product_id not in item_similarity_df.columns:
            return []
        
        similar_scores = item_similarity_df[product_id]
        similar_products = similar_scores.sort_values(ascending=False).head(top_n + 1).index.tolist()
        return [p for p in similar_products if p != product_id][:top_n]
    except Exception as e:
        print(f"Collaborative filtering error: {e}")
        return []

def get_content_bundles(product_id, top_n=5):
    """Get recommendations using content-based filtering"""
    try:
        if content_similarity_df is None or product_id not in content_similarity_df.index:
            return []
        
        similar_scores = content_similarity_df.loc[product_id]
        similar_products = similar_scores.sort_values(ascending=False).head(top_n + 1).index.tolist()
        return [p for p in similar_products if p != product_id][:top_n]
    except Exception as e:
        print(f"Content-based error: {e}")
        return []

def get_graph_bundles(product_id, top_n=5):
    """Get recommendations using graph co-purchase analysis"""
    try:
        if G is None or product_id not in G:
            return []
        
        neighbors = list(G[product_id].items())
        neighbors.sort(key=lambda x: x[1].get('weight', 0), reverse=True)
        
        return [neighbor[0] for neighbor in neighbors[:top_n]]
    except Exception as e:
        print(f"Graph co-purchase error: {e}")
        return []

def get_temporal_trends(df, product_id, window_days=90, top_n=10):
    try:
        if df is None or df.empty:
            return []

        local_data = df.copy()

        if not pd.api.types.is_datetime64_any_dtype(local_data['order_date']):
            local_data['order_date'] = pd.to_datetime(local_data['order_date'], errors='coerce')

        local_data = local_data.dropna(subset=['order_date'])
        if local_data.empty:
            return []

        recent_date = local_data['order_date'].max()
        cutoff_date = recent_date - pd.Timedelta(days=window_days)

        recent_orders = local_data[local_data['order_date'] >= cutoff_date]
        if recent_orders.empty:
            recent_orders = local_data

        product_orders = recent_orders[
            recent_orders['product_id'] == product_id
        ]['order_id'].unique()

        if len(product_orders) == 0:
            return []

        co_purchase_counts = {}
        for oid in product_orders:
            products = local_data[local_data['order_id'] == oid]['product_id']
            for p in products:
                if p != product_id:
                    co_purchase_counts[p] = co_purchase_counts.get(p, 0) + 1

        sorted_products = sorted(co_purchase_counts.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_products[:top_n]]

    except Exception as e:
        print(f"Temporal trend error for product {product_id}: {e}")
        return []


def filter_by_price_compatibility(base_product_id, recommendations, price_tolerance=0.3):
    """Filter recommendations based on price compatibility"""
    try:
        if base_product_id not in products_df.index:
            return recommendations
        
        base_price = products_df.loc[base_product_id, 'price']
        
        filtered = []
        for rec in recommendations:
            if rec in products_df.index:
                rec_price = products_df.loc[rec, 'price']
                price_ratio = abs(rec_price - base_price) / max(base_price, 1)
                if price_ratio <= price_tolerance:
                    filtered.append(rec)
        
        return filtered
    except:
        return recommendations

def ensure_diversity(recommendations, similarity_threshold=0.7):
    """Ensure recommendations are diverse (not too similar)"""
    if len(recommendations) <= 1:
        return recommendations
    
    try:
        if content_similarity_df is None:
            return recommendations
        
        feature_vectors = []
        valid_recs = []
        
        for rec in recommendations:
            if rec in content_similarity_df.index:
                feature_vectors.append(content_similarity_df.loc[rec].values)
                valid_recs.append(rec)
        
        if len(feature_vectors) <= 1:
            return recommendations
        
        diverse_recs = [valid_recs[0]]
        
        for i in range(1, len(valid_recs)):
            max_similarity = 0
            for j in range(len(diverse_recs)):
                vec_i = feature_vectors[i]
                vec_j = feature_vectors[j]
                sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-8)
                max_similarity = max(max_similarity, sim)
            
            if max_similarity < similarity_threshold:
                diverse_recs.append(valid_recs[i])
        
        return diverse_recs
    except:
        return recommendations

class EnsembleBundleRecommender:
    """Ensemble model combining all recommendation approaches"""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models}
        
    def get_recommendations(self, product_id, top_n=10):
        all_recommendations = {}
        
        for model_name, model_func in self.models.items():
            try:
                recommendations = model_func(product_id, top_n * 2)
                weight = self.weights.get(model_name, 1.0)
                
                for i, rec in enumerate(recommendations):
                    if rec not in all_recommendations:
                        all_recommendations[rec] = 0
                    all_recommendations[rec] += weight * (1.0 / (i + 1))
            except Exception as e:
                print(f"Model {model_name} error: {e}")
                continue
        
        sorted_recs = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for rec, score in sorted_recs:
            if rec != product_id and rec not in result:
                result.append(rec)
            if len(result) >= top_n:
                break
        
        return result

# Create ensemble model
ensemble_models = {
    'association': get_association_bundles,
    'collaborative': get_collaborative_bundles,
    'content': get_content_bundles,
    'graph': get_graph_bundles
    # 'temporal': get_temporal_trends
}

ensemble_recommender = EnsembleBundleRecommender(
    ensemble_models,
    weights={'association': 0.25, 'collaborative': 0.25, 'content': 0.20, 'graph': 0.20, 'temporal': 0.10}
)

def get_bundle_details(product_ids):
    """Get detailed information for a list of product IDs"""
    details = []
    for pid in product_ids:
        if pid in products_df.index:
            product = products_df.loc[pid]
            details.append({
                'id': pid,
                'name': product['product_name'],
                'category': product['category'],
                'price': product['price'],
                'brand': product['brand'],
                'features': product.get('features_clean', product.get('features', 'No features available'))
            })
    return details

def decode_category(cat_code):
    """Decode category code to name - IMPROVED"""
    try:
        # Try to get category name from products_df first
        if isinstance(cat_code, (int, float, np.integer, np.floating)):
            cat_code = int(cat_code)
        
        # Sample category mapping (add your actual categories)
        category_mapping = {
            0: "Jewelry",
            1: "Appliances", 
            2: "Electronics",
            3: "Home & Kitchen",
            4: "Books",
            5: "Clothing",
            6: "Sports",
            7: "Health",
            8: "Fashion",
            9: "Toys",
            10: "Automotive",
            11: "Furniture"
        }
        
        return category_mapping.get(cat_code, f"Category {cat_code}")
    
    except Exception as e:
        print(f"Category decoding error: {e}")
        return f"Category {cat_code}"

def decode_brand(brand_code):
    """Decode brand code to name"""
    try:
        if isinstance(brand_code, str) and brand_code.replace('.', '').isdigit():
            brand_code = float(brand_code)
        
        if label_encoders and 'brand' in label_encoders:
            mapping = label_encoders['brand']
            if isinstance(mapping, dict):
                for key, value in mapping.items():
                    if value == brand_code:
                        return str(key)
        return str(brand_code)
    except:
        return str(brand_code)
if not isinstance(raw_df, pd.DataFrame):
    raise TypeError(f"Expected data to be DataFrame, got {type(raw_df)}")
if 'product_id' not in raw_df.columns:
    raise KeyError("product_id column missing from data")

def create_temporal_insights(df, product_id, window_days=90):
    """
    Generate comprehensive temporal insights for a product.
    Returns dict with recent co-purchases, monthly trends, and user activity.
    """
    try:
        print("DEBUG df type:", type(df))

        
        # Check if data is available
        if df is None or df.empty:
            return EMPTY_TEMPORAL_INSIGHTS.copy()
        
        # Create a copy to avoid modifying original data
        data_copy = df.copy()
        
        # 1. Recent Co-Purchases (last 90 days by default)
        recent_co_purchases = get_temporal_trends(df, product_id, window_days)

        
        # 2. Monthly Trends for THIS PRODUCT
        data_copy['order_date'] = pd.to_datetime(data_copy['order_date'], errors='coerce')
        data_copy = data_copy.dropna(subset=['order_date'])
        
        if data_copy.empty:
            return {
                'recent_co_purchases': recent_co_purchases,
                'monthly_trends': None,
                'user_activity': None,
                'purchase_frequency': 0,
                'avg_order_value': 0
            }
        
        # Get date range for monthly trends (last 12 months)
        end_date = data_copy['order_date'].max()
        start_date = end_date - pd.DateOffset(months=12)
        
        # Filter data for this product
        product_data = data_copy[data_copy['product_id'] == product_id].copy()
        
        purchase_frequency = 0
        avg_order_value = 0
        
        if not product_data.empty:
            # Calculate overall stats
            purchase_frequency = product_data['order_id'].nunique()
            avg_order_value = product_data['item_total'].mean() if len(product_data) > 0 else 0
            
            # Filter to last 12 months for trends
            recent_product_data = product_data[product_data['order_date'] >= start_date]
            
            if not recent_product_data.empty:
                # Group by month
                recent_product_data['month'] = recent_product_data['order_date'].dt.to_period('M').astype(str)
                
                monthly_trends = recent_product_data.groupby('month').agg({
                    'order_id': 'nunique',
                    'item_total': 'sum',
                    'quantity': 'sum'
                }).reset_index()
                
                monthly_trends = monthly_trends.sort_values('month')
                monthly_trends.columns = ['month', 'total_orders', 'total_revenue', 'total_quantity']
            else:
                monthly_trends = None
        else:
            monthly_trends = None
        
        # 3. User Activity
        product_purchases = data_copy[data_copy['product_id'] == product_id]
        
        if not product_purchases.empty:
            # Get user activity for this product
            user_activity = product_purchases.groupby('user_id').agg({
                'order_date': 'max',
                'order_id': 'count',
                'item_total': 'sum'
            }).reset_index()
            
            user_activity = user_activity.rename(columns={
                'order_date': 'last_purchase',
                'order_id': 'purchase_count',
                'item_total': 'total_spent'
            })
            
            user_activity = user_activity.sort_values('last_purchase', ascending=False).head(10)
            user_activity['last_purchase'] = user_activity['last_purchase'].dt.strftime('%Y-%m-%d')
        else:
            user_activity = None
        
        return {
            'recent_co_purchases': recent_co_purchases,
            'monthly_trends': monthly_trends,
            'user_activity': user_activity,
            'purchase_frequency': purchase_frequency,
            'avg_order_value': avg_order_value
        }
    
    except Exception as e:
        print(f"Temporal insights error for product {product_id}: {e}")
        return {
            'recent_co_purchases': [],
            'monthly_trends': None,
            'user_activity': None,
            'purchase_frequency': 0,
            'avg_order_value': 0
        }

# ---------------------------
# Sidebar Configuration
# ---------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%); border-radius: 15px; margin-bottom: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1);">
            <h2 style="color: #f1f5f9; margin: 0;">⚙️ Configuration</h2>
            <p style="color: rgba(241, 245, 249, 0.8); margin: 5px 0 0 0;">Fine-tune your recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Product selection with search capability
    # Product selection with validation
    st.markdown("### 🎯 Select Product")
    if raw_df is not None:
        # Get unique products with purchase counts
        product_stats = raw_df.groupby('product_name').agg({
            'order_id': 'nunique',
            'item_total': 'sum'
        }).reset_index()
        
        # Sort by popularity
        product_stats = product_stats.sort_values('order_id', ascending=False)
        
        # Create display strings
        product_stats['display'] = product_stats.apply(
            lambda row: f"{row['product_name']} | 🛒 {row['order_id']} orders | 💰 ${row['item_total']:.0f}",
            axis=1
        )
        
        product_options = product_stats['display'].tolist()
        
        # Default to first product (most popular)
        selected_display = st.selectbox(
            "Select product (sorted by popularity)",
            product_options,
            index=0,
            help="Products with more orders have better recommendation data"
        )
        
        # Extract product name from display string
        product_name = selected_display.split(' | ')[0]
    else:
        product_name = st.selectbox("Select Product", ["No data available"])
    
    # User selection with stats
    st.markdown("### 👤 Select User")
    with st.expander("User Details", expanded=True):
        if raw_df is not None:
            user_stats = raw_df.groupby('user_id').agg({
                'order_id': 'nunique',
                'item_total': 'sum',
                'user_age': 'first'
            }).reset_index()
            
            user_stats['display'] = user_stats.apply(
                lambda row: f"{row['user_id'][:15]}... | 🛒 {row['order_id']} orders | 💰 ${row['item_total']:.0f}",
                axis=1
            )
            
            user_display_list = user_stats['display'].tolist()
            selected_display = st.selectbox(
                "Choose user",
                user_display_list,
                help="Users with more purchase history yield better recommendations"
            )
            
            selected_user = user_stats[user_stats['display'] == selected_display].iloc[0]
            user_id = selected_user['user_id']
            user_age = selected_user['user_age']
        else:
            user_id = "No user data"
            user_age = None
    
    # Algorithm configuration
    st.markdown("### ⚡ Algorithm Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        use_ensemble = st.checkbox("Ensemble Model", value=True, 
                                  help="Combine all recommendation algorithms")
    with col2:
        max_bundles = st.slider("Max Bundles", 1, 10, 5)
    
    # Advanced filters
    st.markdown("### 🎚️ Advanced Filters")
    price_tolerance = st.slider("Price Tolerance (%)", 0, 100, 30, 
                               help="Maximum price difference between products in bundle")
    diversity_level = st.slider("Diversity Level", 0.0, 1.0, 0.7, 0.1,
                               help="How diverse products should be within bundle")
    temporal_window = st.slider("Recent Trends (days)", 7, 365, 90,
                               help="Consider purchases from last N days")
    
    # Recommendation button
    if st.button("🎯 Generate Smart Bundles", use_container_width=True, type="primary"):
        st.session_state['generate_clicked'] = True
        st.session_state['product_name'] = product_name
        st.session_state['user_id'] = user_id
    else:
        if 'generate_clicked' not in st.session_state:
            st.session_state['generate_clicked'] = False

# ---------------------------
# Main Content Area
# ---------------------------
# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Status indicators
if raw_df is not None and products_df is not None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="status-box">✅ Data Loaded</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="status-box">✅ Models Ready</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="status-box">✅ Analytics Active</div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="status-box">📊 {len(products_df)} Products</div>', unsafe_allow_html=True)

st.markdown("---")

if st.session_state.get('generate_clicked', False):
    product_name = st.session_state.get('product_name')
    user_id = st.session_state.get('user_id')
    
    if product_name and user_id and raw_df is not None:
        # Get product ID
        # Raw product_id (for temporal analytics)
        raw_product_id = raw_product_lookup.get(product_name)

        if raw_product_id is None:
            st.error(f"Product '{product_name}' not found in raw data")
            st.stop()

        # Encoded / model product_id (for recommender)
        model_product_id = products_df[
            products_df['product_name'] == product_name
        ].index[0]


        # Show debug info
        # st.info(f"Selected: {product_name} (ID: {product_id[:8]}...) - Found {len(product_matches)} matching records")


        # Show loading animation
        with st.spinner('🔍 Analyzing patterns and generating recommendations...'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            steps = [
                "Loading user profile...",
                "Analyzing purchase history...",
                "Computing association rules...",
                "Finding similar users...",
                "Analyzing product features...",
                "Building co-purchase graph...",
                "Applying filters...",
                "Generating final bundles..."
            ]
            
            for i, step in enumerate(steps):
                time.sleep(0.3)
                progress_bar.progress((i + 1) * (100 // len(steps)))
                status_text.text(f"🔄 {step}")
        
        status_text.empty()
        progress_bar.empty()
        
        # Get individual model recommendations
        st.markdown("### 📊 Individual Model Recommendations")
        
        cols = st.columns(5)
        model_results = {}
        
        with cols[0]:
            st.markdown("**🤝 Association Rules**")
            assoc_recs = get_association_bundles(model_product_id, 5)
            model_results['association'] = assoc_recs
            if assoc_recs:
                for rec in assoc_recs[:3]:
                    if rec in products_df.index:
                        st.markdown(f"• {products_df.loc[rec, 'product_name'][:20]}...")
            else:
                st.markdown("• No recommendations")
        
        with cols[1]:
            st.markdown("**👥 Collaborative**")
            collab_recs = get_collaborative_bundles(model_product_id, 5)
            model_results['collaborative'] = collab_recs
            if collab_recs:
                for rec in collab_recs[:3]:
                    if rec in products_df.index:
                        st.markdown(f"• {products_df.loc[rec, 'product_name'][:20]}...")
            else:
                st.markdown("• No recommendations")
        
        with cols[2]:
            st.markdown("**📝 Content-Based**")
            content_recs = get_content_bundles(model_product_id, 5)
            model_results['content'] = content_recs
            if content_recs:
                for rec in content_recs[:3]:
                    if rec in products_df.index:
                        st.markdown(f"• {products_df.loc[rec, 'product_name'][:20]}...")
            else:
                st.markdown("• No recommendations")
        
        with cols[3]:
            st.markdown("**🕸️ Graph Co-Purchase**")
            graph_recs = get_graph_bundles(model_product_id, 5)
            model_results['graph'] = graph_recs
            if graph_recs:
                for rec in graph_recs[:3]:
                    if rec in products_df.index:
                        st.markdown(f"• {products_df.loc[rec, 'product_name'][:20]}...")
            else:
                st.markdown("• No recommendations")
        
        with cols[4]:
            st.markdown("**⏰ Recent Trends**")
            temporal_recs = get_temporal_trends(raw_df, raw_product_id, temporal_window, 5)
            model_results['temporal'] = temporal_recs
            if temporal_recs:
                for rec in temporal_recs[:3]:
                    if rec in products_df.index:
                        st.markdown(f"• {products_df.loc[rec, 'product_name'][:20]}...")
            else:
                st.markdown("• No recent trends")
        
        # Get ensemble recommendations
        st.markdown("---")
        st.markdown("### 🏆 Ensemble Recommendations (Combined Intelligence)")
        
        if use_ensemble:
            ensemble_recs = ensemble_recommender.get_recommendations(model_product_id, max_bundles * 3)
            
            # Apply filters
            filtered_recs = filter_by_price_compatibility(
                model_product_id, 
                ensemble_recs, 
                price_tolerance / 100
            )
            
            diverse_recs = ensure_diversity(filtered_recs, 1 - diversity_level)
            final_recommendations = diverse_recs[:max_bundles]
        else:
            # Use only association rules
            final_recommendations = assoc_recs[:max_bundles]
        
        # Display bundles
        if final_recommendations:
            bundle_details = get_bundle_details(final_recommendations)
            
            # Create metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Bundles", len(bundle_details))
            with col2:
                if bundle_details:
                    avg_price = np.mean([p['price'] for p in bundle_details])
                    st.metric("Avg Price", f"${avg_price:.2f}")
                else:
                    st.metric("Avg Price", "$0.00")
            with col3:
                if bundle_details:
                    unique_categories = len(set(p['category'] for p in bundle_details))
                    st.metric("Categories", unique_categories)
                else:
                    st.metric("Categories", 0)
            with col4:
                if bundle_details and content_similarity_df is not None:
                    similarities = []
                    for p in bundle_details:
                        pid = p['id']
                        if pid in content_similarity_df.index and model_product_id in content_similarity_df.index:
                            sim = content_similarity_df.loc[model_product_id, pid]
                            similarities.append(sim)
                    if similarities:
                        avg_similarity = np.mean(similarities)
                        st.metric("Avg Similarity", f"{avg_similarity:.2%}")
                    else:
                        st.metric("Avg Similarity", "N/A")
                else:
                    st.metric("Avg Similarity", "N/A")
            
            # Display bundles in cards
            st.markdown("#### 🎁 Recommended Bundles")
            if bundle_details:
                bundle_cols = st.columns(min(len(bundle_details), 3))
                
                for idx, bundle in enumerate(bundle_details):
                    col_idx = idx % 3
                    with bundle_cols[col_idx]:
                        st.markdown(f"""
                            <div class="card">
                                <h4>Bundle #{idx + 1}</h4>
                                <p><strong>🎯 {bundle['name']}</strong></p>
                                <p><strong>🏷️ Price:</strong> ${bundle['price']:.2f}</p>
                                <p><strong>📂 Category:</strong> {decode_category(bundle['category'])}</p>
                                <p><strong>🏢 Brand:</strong> {decode_brand(bundle['brand'])}</p>
                                <p><strong>✨ Features:</strong> {str(bundle['features'])[:50]}...</p>
                                <div class="badge badge-ensemble">Rank: {idx + 1}</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Download option
                bundle_df = pd.DataFrame(bundle_details)
                csv = bundle_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations",
                    data=csv,
                    file_name="product_bundles.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.warning("No bundle details available")
        else:
            st.warning("No suitable bundles found. Try adjusting filters.")
        
        # ---------------------------
        # Enhanced Visualizations
        # ---------------------------
        st.markdown("---")
        st.markdown("### 📈 Advanced Analytics Dashboard")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance Metrics", 
            "📈 Model Comparison", 
            "🎯 Product Analysis",
            "🔄 Co-Purchase Network",
            "📅 Temporal Insights"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Model performance comparison
                model_names = list(model_results.keys())
                model_counts = [len(recs) for recs in model_results.values()]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=model_names,
                        y=model_counts,
                        marker_color=['#60a5fa', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444'],
                        text=model_counts,
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Recommendations per Model",
                    xaxis_title="Model",
                    yaxis_title="Number of Recommendations",
                    template="plotly_dark",
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price distribution
                if bundle_details:
                    prices = [b['price'] for b in bundle_details]
                    fig = px.histogram(
                        x=prices,
                        nbins=10,
                        title="Price Distribution of Recommendations",
                        color_discrete_sequence=['#8b5cf6']
                    )
                    fig.update_layout(
                        xaxis_title="Price ($)",
                        yaxis_title="Count",
                        height=400,
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f1f5f9'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No price data available")
        
        with tab2:
            # Model overlap analysis
            all_recs = []
            for model, recs in model_results.items():
                for rec in recs:
                    if rec in products_df.index:
                        all_recs.append({
                            'model': model,
                            'product': products_df.loc[rec, 'product_name'],
                            'price': products_df.loc[rec, 'price'],
                            'category': decode_category(products_df.loc[rec, 'category'])
                        })
            
            if all_recs:
                overlap_df = pd.DataFrame(all_recs)
                
                # Create sunburst chart
                fig = px.sunburst(
                    overlap_df,
                    path=['model', 'category', 'product'],
                    values=[1] * len(overlap_df),
                    title="Model-Category-Product Hierarchy",
                    color='model',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    height=600,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#f1f5f9'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No overlap data available")
        
        with tab3:
            # Product similarity matrix
            try:
                product_ids = [model_product_id] + final_recommendations[:5]
                valid_ids = [pid for pid in product_ids if pid in content_similarity_df.index]
                
                if len(valid_ids) > 1:
                    similarity_matrix = content_similarity_df.loc[valid_ids, valid_ids]
                    
                    fig = px.imshow(
                        similarity_matrix,
                        text_auto='.2f',
                        color_continuous_scale='Viridis',
                        title="Product Similarity Matrix",
                        labels=dict(x="Products", y="Products", color="Similarity")
                    )
                    fig.update_layout(
                        height=500,
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f1f5f9'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough valid products for similarity matrix")
            except:
                st.info("Similarity matrix not available for selected products")
        
        with tab4:
            # Network graph visualization
            try:
                # Get top co-purchased products
                if model_product_id in G:
                    neighbors = list(G[model_product_id].items())[:10]
                    
                    if neighbors:
                        # Create network
                        network = nx.Graph()
                        network.add_node(model_product_id, label=product_name[:20], size=20)
                        
                        for neighbor, data in neighbors:
                            if neighbor in products_df.index:
                                network.add_node(
                                    neighbor, 
                                    label=products_df.loc[neighbor, 'product_name'][:20],
                                    size=10
                                )
                                weight = data.get('weight', 1)
                                network.add_edge(
                                    model_product_id, 
                                    neighbor, 
                                    weight=weight
                                )
                        
                        # Create Plotly network
                        pos = nx.spring_layout(network, seed=42)
                        
                        edge_x = []
                        edge_y = []
                        for edge in network.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=2, color='#60a5fa'),
                            hoverinfo='none',
                            mode='lines'
                        )
                        
                        node_x = []
                        node_y = []
                        node_text = []
                        node_size = []
                        for node in network.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(network.nodes[node]['label'])
                            node_size.append(network.nodes[node].get('size', 10))
                        
                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition="top center",
                            marker=dict(
                                size=node_size,
                                color=['#8b5cf6'] + ['#60a5fa'] * (len(node_x) - 1),
                                line_width=2
                            )
                        )
                        
                        fig = go.Figure(data=[edge_trace, node_trace],
                                       layout=go.Layout(
                                           title='Co-Purchase Network',
                                           showlegend=False,
                                           hovermode='closest',
                                           margin=dict(b=20, l=5, r=5, t=40),
                                           height=500,
                                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                           paper_bgcolor='rgba(0,0,0,0)',
                                           plot_bgcolor='rgba(0,0,0,0)',
                                           font_color='#f1f5f9'
                                       ))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No co-purchase network data available")
                else:
                    st.info("Product not found in co-purchase network")
            except Exception as e:
                st.info(f"Network visualization not available: {str(e)[:100]}")
        
        with tab5:
            st.markdown("### 📅 Temporal Insights & Purchase Patterns")
            
            try:
                # Create temporal insights
                st.write("DEBUG raw_product_id:", raw_product_id)


                temporal_insights = create_temporal_insights(raw_df, raw_product_id, temporal_window)
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                # Safely get values with defaults
                purchase_freq = temporal_insights.get('purchase_frequency', 0)
                avg_order_val = temporal_insights.get('avg_order_value', 0)
                recent_co_purchases = temporal_insights.get('recent_co_purchases', [])
                
                with col1:
                    st.metric("Total Purchases", purchase_freq)
                with col2:
                    st.metric("Avg Order Value", f"${avg_order_val:.2f}")
                with col3:
                    st.metric("Recent Co-Purchases", len(recent_co_purchases))
                with col4:
                    st.metric("Analysis Window", f"{temporal_window} days")
                
                # Display recent co-purchases
                st.markdown("#### 🕒 Recent Co-Purchase Trends")
                if recent_co_purchases:
                    recent_data = []
                    for pid in recent_co_purchases[:8]:
                        if pid in products_df.index:
                            recent_data.append({
                                'Product': products_df.loc[pid, 'product_name'][:50],
                                'Category': decode_category(products_df.loc[pid, 'category']),
                                'Price': f"${products_df.loc[pid, 'price']:.2f}",
                                'Co-purchases': 'Frequent'
                            })
                    
                    if recent_data:
                        recent_df = pd.DataFrame(recent_data)
                        st.dataframe(recent_df, use_container_width=True, height=300)
                    else:
                        st.info("Co-purchased products not found in product catalog")
                else:
                    st.info(f"No recent co-purchase trends found for '{product_name}' in the last {temporal_window} days")
                
                # Display monthly trends
                st.markdown("#### 📈 Monthly Purchase Trends")
                monthly_trends = temporal_insights.get('monthly_trends')
                
                if monthly_trends is not None and not monthly_trends.empty:
                    # Create visualization
                    fig = go.Figure()
                    
                    # Add bars for quantity
                    fig.add_trace(go.Bar(
                        x=monthly_trends['month'],
                        y=monthly_trends['total_quantity'],
                        name='Quantity Sold',
                        marker_color='#60a5fa',
                        text=monthly_trends['total_quantity'],
                        textposition='auto',
                    ))
                    
                    # Add line for orders
                    fig.add_trace(go.Scatter(
                        x=monthly_trends['month'],
                        y=monthly_trends['total_orders'],
                        mode='lines+markers',
                        name='Number of Orders',
                        line=dict(color='#8b5cf6', width=3),
                        marker=dict(size=8),
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        title=f"📊 Purchase Trends for '{product_name[:30]}...'",
                        xaxis_title="Month",
                        yaxis_title="Quantity Sold",
                        yaxis2=dict(
                            title="Number of Orders",
                            overlaying='y',
                            side='right'
                        ),
                        height=400,
                        template='plotly_dark',
                        hovermode='x unified',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f1f5f9'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.markdown("##### Monthly Data")
                    st.dataframe(
                        monthly_trends,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info(f"No monthly purchase trend data available for '{product_name}'")
                
                # Display user activity
                st.markdown("#### 👥 Recent User Activity")
                user_activity = temporal_insights.get('user_activity')
                
                if user_activity is not None and not user_activity.empty:
                    # Create user activity chart
                    fig = px.bar(
                        user_activity.head(5),
                        x='user_id',
                        y='purchase_count',
                        title='Top 5 Users by Purchase Count',
                        labels={'purchase_count': 'Number of Purchases', 'user_id': 'User ID'},
                        color='purchase_count',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        height=300,
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#f1f5f9',
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show user activity table
                    st.markdown("##### User Purchase Details")
                    display_df = user_activity.copy()
                    
                    # Rename columns for display
                    if 'total_spent' in display_df.columns:
                        display_df = display_df.rename(columns={
                            'user_id': 'User ID',
                            'last_purchase': 'Last Purchase',
                            'purchase_count': 'Purchase Count',
                            'total_spent': 'Total Spent'
                        })
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Total Spent": st.column_config.NumberColumn(format="$%.2f")
                            }
                        )
                    else:
                        # If column names are different
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No recent user activity data available for '{product_name}'")
                    
            except Exception as e:
                st.error(f"Error loading temporal insights: {str(e)[:100]}")
                st.info("Temporal insights are not available for this product")
        
        # ---------------------------
        # Business Insights
        # ---------------------------
        st.markdown("---")
        st.markdown("### 💡 Business Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="card">
                    <h4>📈 Market Opportunities</h4>
                    <p>• <strong>Cross-selling potential:</strong> Products in similar price range show high co-purchase rates</p>
                    <p>• <strong>Seasonal trends:</strong> Consider time-based bundling strategies</p>
                    <p>• <strong>Customer segmentation:</strong> Different user profiles prefer different bundle types</p>
                    <p>• <strong>Inventory optimization:</strong> Stock products that frequently appear together</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="card">
                    <h4>🎯 Actionable Strategies</h4>
                    <p>• <strong>Personalized offers:</strong> Use user purchase history for targeted recommendations</p>
                    <p>• <strong>Dynamic pricing:</strong> Adjust bundle prices based on demand patterns</p>
                    <p>• <strong>Promotional bundles:</strong> Create special offers for frequently co-purchased items</p>
                    <p>• <strong>Customer retention:</strong> Use insights to improve customer satisfaction</p>
                </div>
            """, unsafe_allow_html=True)
        
    else:
        st.warning("Please select both product and user")

else:
    # Welcome screen with metrics
    st.markdown("""
        <div style="text-align: center; padding: 40px 0;">
            <h2>🚀 Welcome to Smart Bundle Recommender</h2>
            <p style="font-size: 1.2rem; color: #94a3b8; margin: 20px 0;">
                Leverage AI to discover perfect product combinations that increase sales and customer satisfaction
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    if raw_df is not None and products_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products = len(products_df)
            st.metric("📦 Total Products", f"{total_products:,}")
        
        with col2:
            total_users = len(raw_df['user_id'].unique())
            st.metric("👥 Total Users", f"{total_users:,}")
        
        with col3:
            total_orders = len(raw_df['order_id'].unique())
            st.metric("🛒 Total Orders", f"{total_orders:,}")
        
        with col4:
            avg_basket = raw_df.groupby('order_id')['item_total'].sum().mean()
            st.metric("💰 Avg Basket", f"${avg_basket:.2f}")
    else:
        st.warning("Data not loaded properly. Please check your files.")
    
    # Quick start guide
    st.markdown("---")
    st.markdown("### 🎯 How to Get Started")
    
    guide_cols = st.columns(3)
    
    with guide_cols[0]:
        st.markdown("""
            <div class="card">
                <h4>1. Select Product</h4>
                <p>Choose any product from our extensive catalog</p>
            </div>
        """, unsafe_allow_html=True)
    
    with guide_cols[1]:
        st.markdown("""
            <div class="card">
                <h4>2. Choose User Profile</h4>
                <p>Select a user to personalize recommendations</p>
            </div>
        """, unsafe_allow_html=True)
    
    with guide_cols[2]:
        st.markdown("""
            <div class="card">
                <h4>3. Generate Bundles</h4>
                <p>Get AI-powered bundle recommendations</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    st.markdown("### ✨ Key Features")
    
    feature_cols = st.columns(4)
    
    features = [
        ("🤖 AI Ensemble", "Combines 5 algorithms for optimal results"),
        ("📊 Advanced Analytics", "Real-time insights and visualizations"),
        ("🎯 Personalization", "User-specific recommendations"),
        ("⚡ Real-time", "Instant bundle generation")
    ]
    
    for idx, (title, desc) in enumerate(features):
        with feature_cols[idx]:
            st.markdown(f"""
                <div class="card">
                    <h4>{title}</h4>
                    <p style="color: #94a3b8;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        <p>🎯 Smart Product Bundle Recommender | Powered by Ensemble AI | Made with ❤️ by Gautam Sharma</p>
        <p style="font-size: 12px; opacity: 0.8;">Combining Association Rules, Collaborative Filtering, Content-Based Matching & Graph Analytics</p>
    </div>
""", unsafe_allow_html=True)
