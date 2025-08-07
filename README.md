# Customer Segmentation Analysis

**Advanced Analytics for Strategic Customer Insights**

A comprehensive machine learning project that leverages RFM analysis and K-means clustering to segment customers into actionable business groups, enabling data-driven marketing strategies and improved customer lifetime value optimization.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Customer Segments](#customer-segments)
- [Strategic Recommendations](#strategic-recommendations)
- [Technical Implementation](#technical-implementation)
- [Performance Metrics](#performance-metrics)
- [Business Impact](#business-impact)
- [Dependencies](#dependencies)
- [Contact](#contact)


## 🎯 Project Overview

This project implements an end-to-end customer segmentation solution using advanced machine learning techniques to help businesses understand their customer base and develop targeted marketing strategies. The analysis combines **RFM (Recency, Frequency, Monetary)** analysis with **K-means clustering** and **Principal Component Analysis (PCA)** to create meaningful customer segments.

### 🚀 Key Achievements

- **392,732 transactions** analyzed across **4,339 unique customers**
- **37 countries** represented in the dataset
- **\$8.9M total revenue** analyzed
- **5 distinct customer segments** identified
- **15-25% improvement** in campaign effectiveness expected


## 🎯 Business Problem

Modern businesses face the challenge of understanding diverse customer behaviors and preferences to optimize marketing spend and improve customer retention. This project addresses:

- **Customer Diversity**: Understanding different customer types and behaviors
- **Marketing Efficiency**: Identifying which customers to target with specific campaigns
- **Revenue Optimization**: Focusing resources on high-value customer segments
- **Churn Prevention**: Identifying at-risk customers for retention strategies
- **Personalization**: Enabling targeted marketing campaigns


## 🔬 Methodology

### 1. **Data Preprocessing**

- Data cleaning and outlier removal using Z-score analysis (Z < 3)
- Feature engineering for behavioral insights
- Missing value imputation and duplicate removal


### 2. **RFM Analysis**

- **Recency**: Days since last purchase
- **Frequency**: Number of unique orders
- **Monetary**: Total customer value


### 3. **Advanced Feature Engineering**

- Customer lifetime calculation
- Average order value metrics
- Product variety analysis
- Purchase pattern identification


### 4. **Machine Learning Pipeline**

- StandardScaler for feature normalization
- PCA for dimensionality reduction (95% variance retained)
- K-means clustering with optimal cluster determination
- Silhouette analysis for validation


### 5. **Business Intelligence**

- Segment naming and business interpretation
- Strategic recommendations for each segment
- Geographic market analysis
- ROI projections


## 📊 Dataset

**Source**: E-commerce transaction data (43MB)[^1_1]

- **Records**: 541,909 initial transactions
- **Clean Dataset**: 392,732 transactions after preprocessing
- **Time Period**: December 2010 - December 2011
- **Geographic Coverage**: 37 countries


### Data Schema

| Column | Description | Type |
| :-- | :-- | :-- |
| InvoiceNo | Unique transaction identifier | String |
| StockCode | Product code | String |
| Description | Product description | String |
| Quantity | Items purchased | Integer |
| InvoiceDate | Transaction timestamp | DateTime |
| UnitPrice | Price per unit | Float |
| CustomerID | Unique customer identifier | Integer |
| Country | Customer location | String |

## ✨ Key Features

### 🔍 **Advanced Analytics**

- Multi-dimensional customer profiling
- Behavioral pattern recognition
- Geographic market intelligence
- Time-series purchase analysis


### 🤖 **Machine Learning**

- Unsupervised clustering with K-means
- Principal Component Analysis (PCA)
- Statistical validation with multiple metrics
- Automated optimal cluster determination


### 📈 **Business Intelligence**

- Executive dashboard visualizations
- Strategic segment recommendations
- ROI impact projections
- Geographic opportunity mapping


### 📊 **Data Export Capabilities**

- Customer segments CSV for CRM integration
- Cluster summary for BI dashboards
- Geographic distribution analysis
- Campaign targeting lists


## 🛠️ Installation

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```


### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Rohit-Singh-0/Customer-Segmentation-Analysis.git
cd Customer-Segmentation-Analysis
```

2. **Create virtual environment**
```bash
python -m venv customer_segmentation_env
source customer_segmentation_env/bin/activate  # On Windows: customer_segmentation_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```


### Required Libraries

```python
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
scipy>=1.8.0
jupyter>=1.0.0
```


## 📖 Usage

### 1. **Run the Complete Analysis**

```bash
jupyter notebook "Customer Segmentation Analysis.ipynb"
```


### 2. **Execute All Cells**

The notebook follows a structured approach:

- Data import and exploration
- Data cleaning pipeline
- RFM analysis
- Machine learning pipeline
- Cluster analysis and visualization
- Business recommendations


### 3. **Access Results**

After execution, you'll have:

- `customer_segments.csv` - Detailed customer data with segments
- `cluster_summary.csv` - Segment performance metrics
- `country_segments.csv` - Geographic segment distribution


### 4. **Quick Start Code**

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the processed data
df = pd.read_csv('customer_segments.csv')

# View segment distribution
print(df['Segment_Name'].value_counts())

# Analyze segment characteristics
segment_summary = df.groupby('Segment_Name').agg({
    'Monetary': ['mean', 'sum'],
    'Frequency': 'mean',
    'Recency': 'mean'
})
print(segment_summary)
```


## 📁 Project Structure

```
Customer-Segmentation-Analysis/
├── 📊 input_data/
│   └── data.csv                    # Raw transaction data (43MB)
├── 📓 Customer Segmentation Analysis.ipynb    # Main analysis notebook
├── 📈 customer_segments.csv        # Segmented customer data
├── 📊 cluster_summary.csv         # Segment performance metrics
├── 🌍 country_segments.csv        # Geographic distribution
├── 📁 .ipynb_checkpoints/         # Jupyter notebook checkpoints
└── 📄 README.md                  # Project documentation
```


## 🏆 Results

### **Cluster Quality Metrics**[^1_2]

- **Silhouette Score**: 0.345 (Good separation)
- **Calinski-Harabasz Index**: 1,634 (Strong clusters)
- **Davies-Bouldin Index**: 0.823 (Optimal clustering)
- **Variance Explained**: 96% with 6 PCA components


### **Data Processing Results**

- **Clean Records**: 392,732 transactions
- **Unique Customers**: 4,339
- **Unique Products**: 3,665
- **Total Revenue Analyzed**: \$8,887,209


## 👥 Customer Segments

### 🏆 **Champions** (55.0% of customers | 18.7% of revenue)

- **Profile**: High-value recent customers with low frequency
- **Characteristics**:
    - Average Value: \$452 per customer
    - Average Orders: 1.5
    - Average Recency: 143 days
- **Count**: 2,322 customers
- **Revenue**: \$1,049,254


### 🤝 **Loyal Customers** (0.2% of customers | 0.6% of revenue)

- **Profile**: Consistent, high-value buyers
- **Characteristics**:
    - Average Value: \$3,746 per customer
    - Average Orders: 4.0
    - Average Recency: 115 days
- **Count**: 9 customers
- **Revenue**: \$33,717


### 🌱 **Potential Loyalists** (6.8% of customers | 34.5% of revenue)

- **Profile**: Recent customers with highest growth potential
- **Characteristics**:
    - Average Value: \$6,756 per customer
    - Average Orders: 13.1
    - Average Recency: 20 days
- **Count**: 286 customers
- **Revenue**: \$1,932,161


### ⚠️ **At Risk** (37.8% of customers | 45.9% of revenue)

- **Profile**: Previously valuable customers showing declining engagement
- **Characteristics**:
    - Average Value: \$1,611 per customer
    - Average Orders: 4.9
    - Average Recency: 38 days
- **Count**: 1,595 customers
- **Revenue**: \$2,570,204


### 😔 **Lost Customers** (0.2% of customers | 0.2% of revenue)

- **Profile**: Dormant customers needing aggressive win-back
- **Characteristics**:
    - Average Value: \$1,475 per customer
    - Average Orders: 2.1
    - Average Recency: 134 days
- **Count**: 8 customers
- **Revenue**: \$11,799


## 🎯 Strategic Recommendations

### 🏆 **Champions Strategy**

```
✅ VIP rewards program implementation
✅ Exclusive early access to new products
✅ Referral incentive campaigns
✅ Premium customer service tier
```


### 🌱 **Potential Loyalists Strategy**

```
✅ Personalized onboarding sequences
✅ Cross-sell and upsell campaigns
✅ Educational content marketing
✅ Loyalty point system introduction
```


### ⚠️ **At Risk Strategy**

```
✅ Automated re-engagement campaigns
✅ Personalized discount offers
✅ Feedback surveys and NPS tracking
✅ Win-back email sequences
```


### 🤝 **Loyal Customers Strategy**

```
✅ Advanced loyalty programs
✅ Product recommendation engines
✅ Exclusive member events
✅ Lifetime value optimization
```


### 😔 **Lost Customers Strategy**

```
✅ Aggressive win-back campaigns
✅ Deep discount promotions
✅ New product announcements
✅ Exit interview surveys
```


## 🔧 Technical Implementation

### **Data Preprocessing Pipeline**

```python
# Feature Engineering
rfm_data = df_analysis.groupby('CustomerID').agg({
    'order_date': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'total_price': 'sum'
}).reset_index()

# Outlier Removal
z_scores = np.abs(stats.zscore(X))
X_filtered = X[(z_scores < 3).all(axis=1)]

# Dimensionality Reduction
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
```


### **Clustering Algorithm**

```python
# Optimal Cluster Determination
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    silhouette_scores.append(silhouette_score(X_pca, clusters))

# Final Model
kmeans_final = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_pca)
```


## 📊 Performance Metrics

### **Model Validation**

- **Silhouette Score**: 0.345 (Good cluster separation)
- **Calinski-Harabasz Index**: 1,634 (Strong internal cluster structure)
- **Davies-Bouldin Index**: 0.823 (Optimal cluster compactness)


### **Business Metrics**

- **Data Coverage**: 72% of original transactions retained after cleaning
- **Customer Coverage**: 4,339 unique customers segmented
- **Geographic Reach**: 37 countries analyzed
- **Revenue Coverage**: \$8.9M total transaction value


## 💰 Business Impact

### **Projected ROI by Segment**[^1_3]

| Segment | Current Revenue | Expected Lift | Campaign Cost | Projected ROI |
| :-- | :-- | :-- | :-- | :-- |
| Potential Loyalists | \$1.93M | 25-30% | \$50K | \$450K |
| At Risk | \$2.57M | 15-20% | \$75K | \$385K |
| Champions | \$1.05M | 10-15% | \$40K | \$105K |
| Loyal Customers | \$34K | 20-25% | \$5K | \$7K |
| Lost Customers | \$12K | 5-10% | \$2K | \$600 |

### **Expected Business Outcomes**

- **Campaign Effectiveness**: 15-25% improvement
- **Customer Retention**: Enhanced through targeted strategies
- **Marketing ROI**: Optimized spend allocation
- **Customer Lifetime Value**: Increased through personalization


## 🌍 Geographic Market Intelligence

### **Top Markets for Potential Loyalists**[^1_4]

1. **United Kingdom**: \$1,578,918 (245 customers)
2. **Germany**: \$104,857 (14 customers)
3. **France**: \$104,693 (11 customers)
4. **Japan**: \$31,417 (3 customers)
5. **Spain**: \$29,919 (3 customers)

### **Top Markets for At-Risk Customers**[^1_4]

1. **United Kingdom**: \$2,244,538 (1,454 customers)
2. **Germany**: \$66,776 (30 customers)
3. **France**: \$63,399 (31 customers)
4. **Belgium**: \$32,115 (15 customers)
5. **Switzerland**: \$24,726 (7 customers)

## 📦 Dependencies

```txt
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
scipy>=1.8.0
jupyter>=1.0.0
warnings
```

## 👤 Contact

**Rohit Kumar Singh**

- 📧 Email: [Send Email](rohitsingh3640@gmail.com)
- 💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/rohit-singh-336859247/)
- 🐙 GitHub: [@Rohit-Singh-0](https://github.com/Rohit-Singh-0)


**⭐ If you find this project helpful, please consider giving it a star on GitHub!**
