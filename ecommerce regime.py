!pip install hmmlearn xgboost shap lightgbm plotly streamlit joblib -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
import pickle
import joblib
from datetime import datetime, timedelta

# ML & Stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from hmmlearn.hmm import GaussianHMM

# SHAP
import shap

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

print(" All libraries imported successfully")

DATA_PATH = "/content/"  # Change to your path

def load_data(path):
    train      = pd.read_csv(path + "train.csv")
    stores     = pd.read_csv(path + "stores.csv")
    oil        = pd.read_csv(path + "oil.csv")
    holidays   = pd.read_csv(path + "holidays_events.csv")
    transactions = pd.read_csv(path + "transactions.csv")
    print(f"Train: {train.shape} | Stores: {stores.shape} | Oil: {oil.shape}")
    print(f"Holidays: {holidays.shape} | Transactions: {transactions.shape}")
    return train, stores, oil, holidays, transactions

train, stores, oil, holidays, transactions = load_data(DATA_PATH)
print("\nTrain Sample:")
print(train.head(3))

# CONVERT DATE COLUMNS

for df, col in [(train, 'date'), (oil, 'date'), (holidays, 'date'), (transactions, 'date')]:
    df[col] = pd.to_datetime(df[col])

print("Date columns converted")
print(f"Train date range: {train['date'].min()} → {train['date'].max()}")

# Merge stores info
df = train.merge(stores, on='store_nbr', how='left')

# Merge oil prices (forward fill missing oil prices)
oil = oil.set_index('date').reindex(
    pd.date_range(oil['date'].min(), oil['date'].max(), freq='D')
).ffill().bfill().reset_index()
oil.columns = ['date', 'oil_price']
df = df.merge(oil, on='date', how='left')

# Merge transactions
df = df.merge(transactions, on=['date', 'store_nbr'], how='left')

# Process holidays
holidays['is_holiday'] = 1
holidays['is_national_holiday'] = (holidays['locale'] == 'National').astype(int)
holidays['is_regional_holiday'] = (holidays['locale'] == 'Regional').astype(int)
holidays_agg = holidays.groupby('date').agg(
    is_holiday=('is_holiday', 'max'),
    is_national_holiday=('is_national_holiday', 'max'),
    is_regional_holiday=('is_regional_holiday', 'max')
).reset_index()
df = df.merge(holidays_agg, on='date', how='left')

# Fill holiday flags with 0
for col in ['is_holiday', 'is_national_holiday', 'is_regional_holiday']:
    df[col] = df[col].fillna(0).astype(int)

print(f" Merged dataset shape: {df.shape}")
print(df.dtypes)

daily = df.groupby('date').agg(
    total_sales=('sales', 'sum'),
    total_transactions=('transactions', 'sum'),
    avg_oil_price=('oil_price', 'mean'),
    total_onpromotion=('onpromotion', 'sum'),
    is_holiday=('is_holiday', 'max'),
    is_national_holiday=('is_national_holiday', 'max'),
    is_regional_holiday=('is_regional_holiday', 'max'),
    num_stores=('store_nbr', 'nunique')
).reset_index().sort_values('date')

daily['avg_oil_price'] = daily['avg_oil_price'].ffill().bfill()
daily['total_transactions'] = daily['total_transactions'].fillna(daily['total_transactions'].median())

print(f" Daily aggregated data: {daily.shape}")
print(daily.head(3))

print(f"\nMissing values before cleaning:\n{daily.isnull().sum()}")

# Remove negative sales
daily = daily[daily['total_sales'] >= 0].copy()

# Cap extreme outliers at 99.5th percentile
cap_val = daily['total_sales'].quantile(0.995)
daily['total_sales'] = daily['total_sales'].clip(upper=cap_val)

# Fill remaining nulls
daily = daily.ffill().bfill()

print(f"\n After cleaning: {daily.shape}")
print(f"Sales range: {daily['total_sales'].min():.2f} → {daily['total_sales'].max():.2f}")

#feature engineering
df_feat = daily.copy()
df_feat = df_feat.sort_values('date').reset_index(drop=True)

#  Lag Features
for lag in [1, 7, 14, 21, 30, 60, 90]:
    df_feat[f'sales_lag_{lag}'] = df_feat['total_sales'].shift(lag)

#Rolling Statistics
for window in [7, 14, 30, 60]:
    df_feat[f'rolling_mean_{window}'] = df_feat['total_sales'].shift(1).rolling(window).mean()
    df_feat[f'rolling_std_{window}']  = df_feat['total_sales'].shift(1).rolling(window).std()
    df_feat[f'rolling_min_{window}']  = df_feat['total_sales'].shift(1).rolling(window).min()
    df_feat[f'rolling_max_{window}']  = df_feat['total_sales'].shift(1).rolling(window).max()

# Exponential Weighted Mean
df_feat['ewm_7']  = df_feat['total_sales'].shift(1).ewm(span=7).mean()
df_feat['ewm_30'] = df_feat['total_sales'].shift(1).ewm(span=30).mean()

# Calendar Features
df_feat['year']        = df_feat['date'].dt.year
df_feat['month']       = df_feat['date'].dt.month
df_feat['quarter']     = df_feat['date'].dt.quarter
df_feat['week']        = df_feat['date'].dt.isocalendar().week.astype(int)
df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
df_feat['day_of_month']= df_feat['date'].dt.day
df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
df_feat['is_weekend']  = (df_feat['day_of_week'] >= 5).astype(int)
df_feat['is_month_start'] = df_feat['date'].dt.is_month_start.astype(int)
df_feat['is_month_end']   = df_feat['date'].dt.is_month_end.astype(int)
df_feat['is_quarter_end'] = df_feat['date'].dt.is_quarter_end.astype(int)


df_feat['month_sin']       = np.sin(2 * np.pi * df_feat['month'] / 12)
df_feat['month_cos']       = np.cos(2 * np.pi * df_feat['month'] / 12)
df_feat['day_of_week_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
df_feat['day_of_week_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)

# Business /Demand Features
df_feat['promo_per_store']     = df_feat['total_onpromotion'] / df_feat['num_stores']
df_feat['transactions_per_store'] = df_feat['total_transactions'] / df_feat['num_stores']
df_feat['oil_change']          = df_feat['avg_oil_price'].diff()
df_feat['oil_rolling_7']       = df_feat['avg_oil_price'].rolling(7).mean()
df_feat['is_payday']           = df_feat['day_of_month'].isin([15, 16, 28, 29, 30, 31]).astype(int)

# Trend Features
df_feat['sales_trend_7']  = df_feat['rolling_mean_7']  - df_feat['rolling_mean_30']
df_feat['sales_trend_14'] = df_feat['rolling_mean_14'] - df_feat['rolling_mean_30']

# Drop rows with NaN from lag/rolling (first 90 rows)
df_feat = df_feat.dropna().reset_index(drop=True)

FEATURE_COLS = [c for c in df_feat.columns if c not in ['date', 'total_sales']]
print(f" Total features engineered: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Exploratory Data Analysis — Store Sales', fontsize=18, fontweight='bold')

# 1. Daily Sales Trend
axes[0,0].plot(daily['date'], daily['total_sales'], color='steelblue', linewidth=0.8)
axes[0,0].set_title('Daily Total Sales')
axes[0,0].set_xlabel('Date'); axes[0,0].set_ylabel('Sales')

# 2. Monthly Sales Trend
monthly = daily.copy()
monthly['month_year'] = daily['date'].dt.to_period('M')
monthly_avg = monthly.groupby('month_year')['total_sales'].mean()
axes[0,1].plot(range(len(monthly_avg)), monthly_avg.values, marker='o', color='darkorange', linewidth=1.5)
axes[0,1].set_title('Monthly Avg Sales')
axes[0,1].set_xlabel('Month Index')
axes[0,1].set_xticks(range(0, len(monthly_avg), 12))

# 3. Yearly Sales Trend
yearly = daily.groupby(daily['date'].dt.year)['total_sales'].mean()
axes[0,2].bar(yearly.index, yearly.values, color='teal', edgecolor='black')
axes[0,2].set_title('Yearly Avg Sales'); axes[0,2].set_xlabel('Year')

# 4. Sales Distribution
axes[1,0].hist(daily['total_sales'], bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1,0].set_title('Sales Distribution'); axes[1,0].set_xlabel('Sales')

# 5. Day of Week Analysis
dow = daily.copy()
dow['dow'] = daily['date'].dt.dayofweek
dow_avg = dow.groupby('dow')['total_sales'].mean()
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
axes[1,1].bar(days, dow_avg.values, color='coral', edgecolor='black')
axes[1,1].set_title('Avg Sales by Day of Week')

# 6. Promotion Impact
promo_effect = daily.groupby(daily['total_onpromotion'] > 0)['total_sales'].mean()
axes[1,2].bar(['No Promo', 'With Promo'],
              [daily[daily['total_onpromotion']==0]['total_sales'].mean(),
               daily[daily['total_onpromotion']>0]['total_sales'].mean()],
              color=['gray','green'], edgecolor='black')
axes[1,2].set_title('Promotion Impact on Sales')

# 7. Holiday Impact
hol_avg = daily.groupby('is_holiday')['total_sales'].mean()
axes[2,0].bar(['Non-Holiday','Holiday'], hol_avg.values, color=['blue','red'], edgecolor='black')
axes[2,0].set_title('Holiday Impact on Sales')

# 8. Oil Price vs Sales Scatter
axes[2,1].scatter(daily['avg_oil_price'], daily['total_sales'], alpha=0.3, color='brown', s=10)
axes[2,1].set_title('Oil Price vs Sales')
axes[2,1].set_xlabel('Oil Price'); axes[2,1].set_ylabel('Sales')

# 9. Correlation Heatmap
corr_cols = ['total_sales','total_transactions','avg_oil_price','total_onpromotion',
             'is_holiday','is_national_holiday']
corr = daily[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2,2])
axes[2,2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(" EDA plots saved")

# Select features for HMM (use observable market-like signals)
hmm_features = ['total_sales', 'rolling_mean_7', 'rolling_std_7',
                 'total_onpromotion', 'avg_oil_price']
hmm_data = df_feat[hmm_features].copy()

# Standardize
scaler_hmm = StandardScaler()
hmm_scaled = scaler_hmm.fit_transform(hmm_data)

# Fit Gaussian HMM with 3 hidden states
N_STATES = 3
hmm_model = GaussianHMM(
    n_components=N_STATES,
    covariance_type='full',
    n_iter=200,
    random_state=42
)
hmm_model.fit(hmm_scaled)

# Predict hidden states
raw_states = hmm_model.predict(hmm_scaled)
df_feat['raw_state'] = raw_states

# --- Auto-map states by average sales (no manual labeling) ---
state_avg_sales = df_feat.groupby('raw_state')['total_sales'].mean()
state_rank = state_avg_sales.rank()  # 1=lowest, 3=highest

state_map = {}
for state, rank in state_rank.items():
    if rank == 3:
        state_map[state] = 2  # High Demand
    elif rank == 2:
        state_map[state] = 1  # Normal Demand
    else:
        state_map[state] = 0  # Low Demand

df_feat['regime'] = df_feat['raw_state'].map(state_map)
REGIME_NAMES = {0: 'Low Demand', 1: 'Normal Demand', 2: 'High Demand'}
REGIME_COLORS = {0: '#e74c3c', 1: '#f39c12', 2: '#27ae60'}
df_feat['regime_label'] = df_feat['regime'].map(REGIME_NAMES)

print(" HMM fitted and regimes detected")
print("\nRegime Distribution:")
print(df_feat['regime_label'].value_counts())
print("\nAverage Sales by Regime:")
print(df_feat.groupby('regime_label')['total_sales'].mean().round(2))

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Demand Regime Detection — Hidden Markov Model', fontsize=16, fontweight='bold')

colors = df_feat['regime'].map(REGIME_COLORS)

# 1. Sales Timeline Colored by Regime
axes[0,0].scatter(df_feat['date'], df_feat['total_sales'],
                  c=df_feat['regime'].map(REGIME_COLORS), s=8, alpha=0.7)
patches = [mpatches.Patch(color=REGIME_COLORS[i], label=REGIME_NAMES[i]) for i in range(3)]
axes[0,0].legend(handles=patches)
axes[0,0].set_title('Sales Timeline by Demand Regime')
axes[0,0].set_xlabel('Date'); axes[0,0].set_ylabel('Total Sales')

# 2. Regime Scatter: Rolling Mean vs Std
for regime in range(3):
    mask = df_feat['regime'] == regime
    axes[0,1].scatter(df_feat.loc[mask, 'rolling_mean_7'],
                      df_feat.loc[mask, 'rolling_std_7'],
                      c=REGIME_COLORS[regime], label=REGIME_NAMES[regime],
                      s=20, alpha=0.6)
axes[0,1].set_title('Regime Scatter: Rolling Mean vs Std')
axes[0,1].set_xlabel('7-Day Rolling Mean'); axes[0,1].set_ylabel('7-Day Rolling Std')
axes[0,1].legend()

# 3. Regime Distribution
regime_counts = df_feat['regime_label'].value_counts()
axes[1,0].bar(regime_counts.index, regime_counts.values,
              color=[REGIME_COLORS[k] for k in range(3)], edgecolor='black')
axes[1,0].set_title('Regime Distribution (Day Count)')
axes[1,0].set_ylabel('Number of Days')

# 4. Average Sales by Regime
avg_by_regime = df_feat.groupby('regime_label')['total_sales'].mean()
axes[1,1].barh(avg_by_regime.index, avg_by_regime.values,
               color=[REGIME_COLORS[0], REGIME_COLORS[1], REGIME_COLORS[2]],
               edgecolor='black')
axes[1,1].set_title('Average Sales by Regime')
axes[1,1].set_xlabel('Average Total Sales')

plt.tight_layout()
plt.savefig('regime_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Regime plots saved")

# Get the properly mapped transition matrix
trans_mat_raw = hmm_model.transmat_
# Remap rows/cols to match our 0=Low,1=Normal,2=High mapping
# Build new ordering based on state_map
old_to_new = state_map  # {old_state: new_label}
new_order = [k for k, v in sorted(old_to_new.items(), key=lambda x: x[1])]

# Reorder transition matrix
trans_mat = trans_mat_raw[np.ix_(new_order, new_order)]
regime_labels_order = [REGIME_NAMES[i] for i in range(3)]

print("=" * 55)
print("REGIME TRANSITION PROBABILITY MATRIX")
print("=" * 55)
trans_df = pd.DataFrame(trans_mat, index=regime_labels_order, columns=regime_labels_order)
print(trans_df.round(4))
print()
for i, name in enumerate(regime_labels_order):
    stay_prob = trans_mat[i, i]
    print(f"  {name}: {stay_prob*100:.1f}% probability of staying in same regime")

# Plot transition heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(trans_df, annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label': 'Transition Probability'})
plt.title('Regime Transition Probability Heatmap', fontsize=14, fontweight='bold')
plt.ylabel('From Regime'); plt.xlabel('To Regime')
plt.tight_layout()
plt.savefig('transition_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("TRAINING REGIME-SPECIFIC XGBOOST MODELS")


TARGET = 'total_sales'
FEAT_COLS = [c for c in df_feat.columns if c not in
             ['date', 'total_sales', 'regime', 'raw_state', 'regime_label']]

regime_models  = {}
regime_scalers = {}
regime_data    = {}

for regime_id in [0, 1, 2]:
    mask = df_feat['regime'] == regime_id
    regime_df = df_feat[mask].copy()
    regime_data[regime_id] = regime_df

    X = regime_df[FEAT_COLS]
    y = regime_df[TARGET]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X, y)
    regime_models[regime_id] = model

    preds = model.predict(X)
    mae  = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2   = r2_score(y, preds)

    print(f"\n  Model {['C (Low)','B (Normal)','A (High)'][regime_id]} — {REGIME_NAMES[regime_id]}")
    print(f"  Samples: {len(regime_df)} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

print("\nAll 3 regime-specific models trained")

print("WALK-FORWARD VALIDATION (EXPANDING WINDOW)")

# Define fold boundaries
wf_splits = [
    ('2013-2016', '2013-01-01', '2016-12-31', '2017-01-01', '2017-12-31'),
    ('2013-2017', '2013-01-01', '2017-12-31', '2018-01-01', '2018-12-31'),
]

# Filter to available date range
max_date = df_feat['date'].max()
min_date = df_feat['date'].min()
print(f"Data range: {min_date.date()} → {max_date.date()}")

# Dynamic splits based on actual data
years = sorted(df_feat['date'].dt.year.unique())
valid_splits = []
for i in range(1, len(years) - 1):
    train_end = f"{years[i]}-12-31"
    test_start = f"{years[i+1]}-01-01"
    test_end   = f"{years[i+1]}-12-31"
    label = f"{years[0]}-{years[i]}"
    valid_splits.append((label, f"{years[0]}-01-01", train_end, test_start, test_end))

wf_results = []

for (fold_label, tr_start, tr_end, te_start, te_end) in valid_splits:
    # Filter
    tr_mask = (df_feat['date'] >= tr_start) & (df_feat['date'] <= tr_end)
    te_mask = (df_feat['date'] >= te_start) & (df_feat['date'] <= te_end)

    train_fold = df_feat[tr_mask]
    test_fold  = df_feat[te_mask]

    if len(test_fold) < 10:
        continue

    # Train a single XGBoost on this fold (global model for WF)
    wf_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1, verbosity=0)
    wf_model.fit(train_fold[FEAT_COLS], train_fold[TARGET])
    preds = wf_model.predict(test_fold[FEAT_COLS])
    y_true = test_fold[TARGET].values

    mae  = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    r2   = r2_score(y_true, preds)
    mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-6))) * 100

    wf_results.append({
        'Fold': fold_label,
        'Train Size': len(train_fold),
        'Test Size': len(test_fold),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R²': round(r2, 4),
        'MAPE (%)': round(mape, 2)
    })
    print(f"  Fold {fold_label}: MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f} | MAPE={mape:.2f}%")

wf_df = pd.DataFrame(wf_results)
print("\n Walk-Forward Validation Summary:")
print(wf_df.to_string(index=False))

# Plot WF metrics
if len(wf_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Walk-Forward Validation Performance', fontsize=14, fontweight='bold')
    metrics = ['MAE', 'RMSE', 'R²', 'MAPE (%)']
    colors_wf = ['steelblue', 'darkorange', 'green', 'red']
    for ax, metric, color in zip(axes.flatten(), metrics, colors_wf):
        ax.bar(wf_df['Fold'], wf_df[metric], color=color, edgecolor='black')
        ax.set_title(f'{metric} by Fold')
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('walkforward_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

print("SHAP EXPLAINABILITY PER REGIME")

shap_values_dict = {}
shap_explainers  = {}

for regime_id in [0, 1, 2]:
    print(f"\n  Computing SHAP for {REGIME_NAMES[regime_id]}...")
    X_regime = regime_data[regime_id][FEAT_COLS]

    # Use TreeExplainer
    explainer = shap.TreeExplainer(regime_models[regime_id])
    shap_vals = explainer.shap_values(X_regime)
    shap_values_dict[regime_id] = shap_vals
    shap_explainers[regime_id]  = explainer

    # SHAP Summary Plot
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_vals, X_regime, max_display=15,
                      title=f"SHAP Summary — {REGIME_NAMES[regime_id]}",
                      show=False)
    plt.tight_layout()
    plt.savefig(f'shap_summary_regime_{regime_id}.png', dpi=120, bbox_inches='tight')
    plt.show()

    # SHAP Bar Plot (mean abs)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_regime, plot_type='bar', max_display=15,
                      show=False)
    plt.title(f"SHAP Feature Importance — {REGIME_NAMES[regime_id]}")
    plt.tight_layout()
    plt.savefig(f'shap_bar_regime_{regime_id}.png', dpi=120, bbox_inches='tight')
    plt.show()

print("SHAP analysis complete for all regimes")

print("AUTOMATED BUSINESS INSIGHTS")

insights = {}

# 1. Promotion influence per regime
for regime_id in [0, 1, 2]:
    df_r = regime_data[regime_id]
    X_r  = df_r[FEAT_COLS]
    sv   = shap_values_dict[regime_id]
    promo_idx = FEAT_COLS.index('total_onpromotion')
    promo_shap = np.abs(sv[:, promo_idx]).mean()
    insights.setdefault('promo_shap', {})[REGIME_NAMES[regime_id]] = promo_shap

# 2. Oil influence per regime
for regime_id in [0, 1, 2]:
    sv = shap_values_dict[regime_id]
    oil_idx = FEAT_COLS.index('avg_oil_price')
    oil_shap = np.abs(sv[:, oil_idx]).mean()
    insights.setdefault('oil_shap', {})[REGIME_NAMES[regime_id]] = oil_shap

# 3. Holiday influence per regime
for regime_id in [0, 1, 2]:
    sv = shap_values_dict[regime_id]
    hol_idx = FEAT_COLS.index('is_holiday')
    hol_shap = np.abs(sv[:, hol_idx]).mean()
    insights.setdefault('holiday_shap', {})[REGIME_NAMES[regime_id]] = hol_shap

# Print insights
promo_high   = insights['promo_shap']['High Demand']
promo_normal = insights['promo_shap']['Normal Demand']
promo_low    = insights['promo_shap']['Low Demand']
ratio_high   = promo_high / (promo_low + 1e-6)

print(f"\n🔍 PROMOTION INSIGHTS:")
print(f"  High Demand regime:   SHAP promotion effect = {promo_high:.2f}")
print(f"  Normal Demand regime: SHAP promotion effect = {promo_normal:.2f}")
print(f"  Low Demand regime:    SHAP promotion effect = {promo_low:.2f}")
print(f"  → Promotions are {ratio_high:.1f}x more influential during High Demand vs Low Demand")

print(f"\n HOLIDAY INSIGHTS:")
for k, v in insights['holiday_shap'].items():
    print(f"  {k}: SHAP holiday effect = {v:.4f}")
max_hol_regime = max(insights['holiday_shap'], key=insights['holiday_shap'].get)
print(f"  → Holiday effects are strongest during {max_hol_regime} regime")

print(f"\n OIL PRICE INSIGHTS:")
for k, v in insights['oil_shap'].items():
    print(f"  {k}: SHAP oil effect = {v:.4f}")
max_oil_regime = max(insights['oil_shap'], key=insights['oil_shap'].get)
print(f"  → Oil price most impacts the {max_oil_regime} regime")

# Business insight comparison chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Business Insights: Demand Driver Influence by Regime', fontsize=14, fontweight='bold')

for ax, (key, title) in zip(axes, [
    ('promo_shap', 'Promotion Influence'),
    ('holiday_shap', 'Holiday Influence'),
    ('oil_shap', 'Oil Price Influence')
]):
    vals = [insights[key][REGIME_NAMES[i]] for i in range(3)]
    bars = ax.bar(list(REGIME_NAMES.values()), vals,
                  color=[REGIME_COLORS[i] for i in range(3)], edgecolor='black')
    ax.set_title(f'{title} (Mean |SHAP|)')
    ax.set_ylabel('Mean |SHAP Value|')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('business_insights.png', dpi=150, bbox_inches='tight')
plt.show()

print("INVENTORY INTELLIGENCE ENGINE")

def get_inventory_recommendation(predicted_sales, current_regime, onpromotion, is_holiday):
    """Rule-based inventory engine using forecasts + regime context"""

    # Base thresholds
    HIGH_THRESH   = df_feat['total_sales'].quantile(0.75)
    NORMAL_THRESH = df_feat['total_sales'].quantile(0.40)

    urgency_score = 0

    # Regime factor
    if current_regime == 2:     urgency_score += 3  # High Demand
    elif current_regime == 1:   urgency_score += 1  # Normal
    else:                        urgency_score -= 1  # Low

    # Sales forecast factor
    if predicted_sales > HIGH_THRESH:   urgency_score += 2
    elif predicted_sales > NORMAL_THRESH: urgency_score += 1
    else:                                urgency_score -= 1

    # Promotion boost
    if onpromotion > df_feat['total_onpromotion'].quantile(0.75):
        urgency_score += 2

    # Holiday boost
    if is_holiday:
        urgency_score += 1

    # Decision
    if urgency_score >= 4:
        return 'INCREASE INVENTORY', 'high', urgency_score
    elif urgency_score >= 1:
        return 'MAINTAIN INVENTORY', 'medium', urgency_score
    else:
        return 'REDUCE INVENTORY', 'low', urgency_score

# Apply to recent data
recent = df_feat.tail(90).copy()

# Predict using the full model trained on all data
all_model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
all_model.fit(df_feat[FEAT_COLS], df_feat[TARGET])
recent['predicted_sales'] = all_model.predict(recent[FEAT_COLS])

recent[['recommendation', 'risk_level', 'urgency_score']] = recent.apply(
    lambda row: pd.Series(get_inventory_recommendation(
        row['predicted_sales'], row['regime'],
        row['total_onpromotion'], row['is_holiday']
    )), axis=1
)

print("\nInventory Recommendation Sample (last 15 days):")
print(recent[['date','predicted_sales','regime_label','recommendation','risk_level']].tail(15).to_string(index=False))

# Distribution
print("\nRecommendation Distribution:")
print(recent['recommendation'].value_counts())

# Inventory dashboard plot
fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.suptitle('Inventory Intelligence Dashboard', fontsize=14, fontweight='bold')

risk_colors = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#27ae60'}
point_colors = recent['risk_level'].map(risk_colors)

axes[0].plot(recent['date'], recent['total_sales'], label='Actual Sales',
             color='steelblue', linewidth=1.5)
axes[0].plot(recent['date'], recent['predicted_sales'], label='Predicted Sales',
             color='orange', linewidth=1.5, linestyle='--')
axes[0].scatter(recent['date'], recent['predicted_sales'],
                c=point_colors, s=40, zorder=5, label='Inventory Risk')
axes[0].legend(); axes[0].set_title('Actual vs Predicted Sales with Inventory Risk')
axes[0].set_ylabel('Sales')

recs = recent['recommendation'].value_counts()
axes[1].barh(recs.index, recs.values, color=['#e74c3c','#f39c12','#27ae60'][:len(recs)], edgecolor='black')
axes[1].set_title('Inventory Recommendation Distribution')
axes[1].set_xlabel('Number of Days')

plt.tight_layout()
plt.savefig('inventory_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("FORECASTING NEXT 30 DAYS")

last_date = df_feat['date'].max()
last_known = df_feat.copy()

future_rows = []
current_df  = last_known.copy()

for i in range(1, 31):
    future_date = last_date + timedelta(days=i)

    # Build feature row by extending the series
    new_row = {}
    new_row['date']              = future_date
    new_row['total_onpromotion'] = current_df['total_onpromotion'].tail(7).mean()
    new_row['avg_oil_price']     = current_df['avg_oil_price'].iloc[-1]
    new_row['total_transactions']= current_df['total_transactions'].tail(7).mean()
    new_row['is_holiday']        = 0
    new_row['is_national_holiday'] = 0
    new_row['is_regional_holiday'] = 0
    new_row['num_stores']        = current_df['num_stores'].iloc[-1]

    # Calendar
    new_row['year']         = future_date.year
    new_row['month']        = future_date.month
    new_row['quarter']      = (future_date.month - 1) // 3 + 1
    new_row['week']         = future_date.isocalendar()[1]
    new_row['day_of_week']  = future_date.weekday()
    new_row['day_of_month'] = future_date.day
    new_row['day_of_year']  = future_date.timetuple().tm_yday
    new_row['is_weekend']   = int(future_date.weekday() >= 5)
    new_row['is_month_start'] = int(future_date.day == 1)
    new_row['is_month_end']   = int(future_date.day in [28,29,30,31])
    new_row['is_quarter_end'] = int(future_date.month in [3,6,9,12] and new_row['is_month_end'])
    new_row['month_sin']       = np.sin(2*np.pi*new_row['month']/12)
    new_row['month_cos']       = np.cos(2*np.pi*new_row['month']/12)
    new_row['day_of_week_sin'] = np.sin(2*np.pi*new_row['day_of_week']/7)
    new_row['day_of_week_cos'] = np.cos(2*np.pi*new_row['day_of_week']/7)
    new_row['is_payday']       = int(future_date.day in [15,16,28,29,30,31])
    new_row['promo_per_store'] = new_row['total_onpromotion'] / max(new_row['num_stores'], 1)
    new_row['transactions_per_store'] = new_row['total_transactions'] / max(new_row['num_stores'], 1)
    new_row['oil_change']      = 0
    new_row['oil_rolling_7']   = new_row['avg_oil_price']

    # Lag features from current_df
    sales_series = current_df['total_sales'].values
    for lag in [1,7,14,21,30,60,90]:
        new_row[f'sales_lag_{lag}'] = sales_series[-lag] if len(sales_series) >= lag else sales_series[-1]

    # Rolling stats
    for w in [7,14,30,60]:
        window_data = sales_series[-w:] if len(sales_series) >= w else sales_series
        new_row[f'rolling_mean_{w}'] = np.mean(window_data)
        new_row[f'rolling_std_{w}']  = np.std(window_data) if len(window_data) > 1 else 0
        new_row[f'rolling_min_{w}']  = np.min(window_data)
        new_row[f'rolling_max_{w}']  = np.max(window_data)

    new_row['ewm_7']  = pd.Series(sales_series).ewm(span=7).mean().iloc[-1]
    new_row['ewm_30'] = pd.Series(sales_series).ewm(span=30).mean().iloc[-1]

    new_row['sales_trend_7']  = new_row['rolling_mean_7']  - new_row['rolling_mean_30']
    new_row['sales_trend_14'] = new_row['rolling_mean_14'] - new_row['rolling_mean_30']

    # Predict
    X_future = pd.DataFrame([new_row])[FEAT_COLS]
    pred_sales = all_model.predict(X_future)[0]
    pred_sales = max(0, pred_sales)  # no negative
    new_row['total_sales'] = pred_sales

    # Predict regime via HMM
    hmm_feats = np.array([[
        pred_sales,
        new_row['rolling_mean_7'],
        new_row['rolling_std_7'],
        new_row['total_onpromotion'],
        new_row['avg_oil_price']
    ]])
    hmm_feat_scaled = scaler_hmm.transform(hmm_feats)
    raw_state_pred = hmm_model.predict(hmm_feat_scaled)[0]
    regime_pred = state_map.get(raw_state_pred, 1)
    new_row['regime'] = regime_pred
    new_row['regime_label'] = REGIME_NAMES[regime_pred]
    new_row['raw_state'] = raw_state_pred

    new_row_series = pd.Series(new_row)
    future_rows.append(new_row)
    current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

future_df = pd.DataFrame(future_rows)

# Plot forecast
fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.suptitle('30-Day Sales Forecast', fontsize=14, fontweight='bold')

# Historical + Forecast
hist_tail = df_feat.tail(90)
axes[0].plot(hist_tail['date'], hist_tail['total_sales'],
             label='Historical', color='steelblue', linewidth=1.5)
axes[0].plot(future_df['date'], future_df['total_sales'],
             label='Forecast', color='darkorange', linewidth=2, linestyle='--')
axes[0].axvline(x=last_date, color='gray', linestyle=':', linewidth=1.5, label='Forecast Start')
axes[0].fill_between(future_df['date'],
                     future_df['total_sales'] * 0.9,
                     future_df['total_sales'] * 1.1,
                     alpha=0.2, color='orange', label='90% CI')
axes[0].legend(); axes[0].set_title('Historical + 30-Day Forecast')
axes[0].set_ylabel('Sales')

# Forecast colored by regime
colors_fut = future_df['regime'].map(REGIME_COLORS)
axes[1].bar(future_df['date'], future_df['total_sales'],
            color=colors_fut, edgecolor='none', alpha=0.8)
patches = [mpatches.Patch(color=REGIME_COLORS[i], label=REGIME_NAMES[i]) for i in range(3)]
axes[1].legend(handles=patches)
axes[1].set_title('30-Day Forecast by Predicted Regime')
axes[1].set_ylabel('Sales')

plt.tight_layout()
plt.savefig('forecast_30days.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n Forecast Summary:")
print(future_df[['date','total_sales','regime_label']].to_string(index=False))

os.makedirs('/content/models', exist_ok=True)

# Save HMM
joblib.dump(hmm_model, '/content/models/hmm_model.pkl')
joblib.dump(scaler_hmm, '/content/models/scaler_hmm.pkl')

# Save regime XGBoost models
for rid in [0, 1, 2]:
    joblib.dump(regime_models[rid], f'/content/models/xgb_regime_{rid}.pkl')

# Save global model
joblib.dump(all_model, '/content/models/xgb_global.pkl')

# Save feature columns
with open('/content/models/feature_cols.pkl', 'wb') as f:
    pickle.dump(FEAT_COLS, f)

# Save state mapping
with open('/content/models/state_map.pkl', 'wb') as f:
    pickle.dump(state_map, f)

# Save 30-day forecast
future_df.to_csv('/content/models/forecast_30days.csv', index=False)
