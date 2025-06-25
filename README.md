# 📊 Sentiment-Driven Trader Behavior Analysis

This project explores the relationship between **cryptocurrency market sentiment** and **trader performance**, using historical trade data and sentiment indices. The objective is to uncover trader behavior patterns under varying emotional market states like *Fear*, *Greed*, and *Extreme Fear*, and to cluster traders based on these behaviors.

---

## 📁 Datasets Used

### 1. Fear & Greed Index Dataset
- Columns: `date`, `value`, `classification`
- Daily market sentiment labels such as "Fear", "Greed", etc.

### 2. Hyperliquid Historical Trader Data
- Key columns: `Account`, `Timestamp IST`, `Closed PnL`, `Size USD`, `Execution Price`, `Fee`, `Trade ID`, etc.
- Detailed order-level trading activity across days and market events.

---

## 🛠️ Workflow Breakdown

### 1. Data Preprocessing
- Parsed and standardized `date` from timestamps in both datasets.
- Merged them using a **left join on the `date` column** to bring sentiment into each trade row.
- Converted `Closed PnL` to numeric for downstream analysis.

### 2. Daily Trader Metrics
Aggregated stats by `Account`, `date`, and `classification` (sentiment):

```python
daily_stats = merged_df.groupby(['Account', 'date', 'classification']).agg({
    'Closed PnL': 'sum',
    'Size USD': 'sum',
    'Fee': 'sum',
    'Execution Price': 'mean',
    'Trade ID': 'count'
}).reset_index()
```

---

### 3. ANOVA – Testing for Sentiment Impact

Used `f_oneway()` to test whether trader performance varied by market sentiment:

```python
from scipy.stats import f_oneway

f_stat, p_val = f_oneway(fear_pnl, extreme_fear_pnl, greed_pnl)
```

- Result: `p < 0.05` → At least one sentiment group has significantly different average PnL.

---

### 4. Exploratory Visualizations

#### Boxplot
Visualized median and spread of `Closed PnL` across sentiments.

#### Violin Plot
Highlighted density, skewness, and variation in trader outcomes for each sentiment.

---

### 5. Sentiment-Based Clustering

#### Feature Creation
Pivoted to create one row per trader, with columns for `avg_PnL_<sentiment>`:

```python
pivot_df = merged_df.pivot_table(
    index='Account',
    columns='classification',
    values='Closed PnL',
    aggfunc='mean'
).add_prefix('avg_PnL_').reset_index()
```

Filled missing values with `0` to handle traders without trades under certain sentiments.

#### Clustering Pipeline

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = pivot_df_filled.drop('Account', axis=1)
scaled = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
pivot_df_filled['cluster'] = kmeans.fit_predict(scaled)
```

---

### 6. Cluster Visualization & Interpretation

#### Pairplot
Compared sentiment-wise PnL metrics and visualized separation of clusters.

#### Radar Chart
Compared **cluster profiles across sentiments** in a spider chart:

```python
import matplotlib.pyplot as plt
import numpy as np
# ... radar chart code ...
```

---

## 🔍 Insights

- **Cluster 0**: Balanced performers with stable PnL across all sentiments.
- **Cluster 1**: Thrive in "Extreme Greed" markets — trend followers.
- **Cluster 2**: Stronger under "Fear" conditions — likely bargain hunters.

These clusters reflect **trader personalities** and reveal opportunities for:
- Adaptive portfolio management
- Targeted strategy design based on sentiment cycles

---

## 📦 Libraries Used
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `scipy`

---

## 🚀 Future Enhancements
- Add sentiment-wise trade count and leverage as clustering dimensions.
- Use `silhouette_score` or `elbow method` to fine-tune cluster count.
- Create an interactive dashboard (e.g. Power BI or Streamlit) for dynamic exploration.

---

By uncovering how sentiment affects performance, this project helps translate behavioral finance into real-world, data-backed trading strategy design.

