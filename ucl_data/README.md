# UCL Analysis Notebooks

Analysis and machine learning experiments on UCL Computer Science academic profiles. This directory contains notebooks
for exploratory analysis, feature engineering, dimensionality reduction, clustering, and supervised learning.

## Contents

| Notebook                                | Purpose                                                         |
|-----------------------------------------|-----------------------------------------------------------------|
| `ucl_data_analysis.ipynb`               | Exploratory data analysis and initial profiling                 |
| `ucl_data_feature_engineering.ipynb`    | Feature engineering (43 features across 10 categories)          |
| `ucl_data_feature_selection.ipynb`      | Dimensionality reduction comparison (correlation filter vs PCA) |
| `ucl_data_clustering.ipynb`             | Clustering experiments (K-means, GMM, hierarchical)             |
| `ucl_data_noise.ipynb`                  | Artificial noise injection and Isolation Forest denoising       |
| `ucl_data_supervised.ipynb`             | Random Forest classification of cluster assignments             |
| `ucl_noise_clustering_comparison.ipynb` | Clustering comparison across raw, noised, and denoised datasets |
| `features.md`                           | Complete feature schema documentation                           |

## Workflow

Run the notebooks in this order:

### 1. Initial analysis

**Notebook:** `ucl_data_analysis.ipynb`

**Input:** `../data/processed/ucl_ml_ready.parquet` (608 profiles, 31 columns)

Performs exploratory data analysis:

- Dataset overview and statistics
- Distribution analysis
- Missing value analysis
- Initial visualisations

### 2. Feature engineering

**Notebook:** `ucl_data_feature_engineering.ipynb`

**Input:** `../data/processed/ucl_ml_ready.parquet`  
**Output:**

- `../data/processed/ucl_features_final.parquet` (503 profiles, 45 features including _index)
- `../data/processed/ucl_features_unscaled.parquet` (for noise experiments and cluster analysis)
- `../data/processed/ucl_identity_mapping.parquet` (identity metadata)

Engineers 43 features + _index across 10 categories:

- Profile completeness (4 binary features)
- Research domains (14 binary features)
- Academic seniority (1 ordinal feature)
- Publication metrics (6 count features)
- Citation metrics (5 features)
- Collaboration metrics (2 features)
- Career-normalised impact (3 derived features)
- Temporal features (2 features)
- Publication composition (4 ratio features)
- Engagement indicators (3 features)

Applies preprocessing:

- Log(x+1) transformation for skewed features
- StandardScaler (z-score normalisation)
- Removes zero-variance features
- Separates identity data into mapping file

### 3. Feature selection / Dimensionality reduction

**Notebook:** `ucl_data_feature_selection.ipynb` (currently named `ucl_data_feature_analysis.ipynb`)

**Input:** `../data/processed/ucl_features_final.parquet`  
**Output:** `../data/processed/ucl_pca_features.parquet` (503 profiles, 13 PCA components + _index)

Compares two dimensionality reduction methods:

- **Correlation filter** (threshold=0.75): Removes 14 features → 30 remaining
- **PCA** (95% variance): Reduces to 13 components

Evaluation metrics:

- Hopkins statistic (clustering tendency)
- Condition number (numerical stability)
- Maximum pairwise correlation (feature redundancy)
- Distance variance (separability)

**Selection:** PCA chosen for superior numerical stability (11.55 vs 31.04 condition number), perfect orthogonality (0.0
vs 0.716 max correlation), and higher distance variance (6.92 vs 2.14).

### 4. Clustering

**Notebook:** `ucl_data_clustering.ipynb`

**Input:** `../data/processed/ucl_pca_features.parquet` (503 profiles, 13 PCA components)

Applies three clustering algorithms:

- **K-means** with elbow method and silhouette analysis (k=2 to 10)
- **Gaussian Mixture Models** (GMM) with BIC criterion
- **Hierarchical clustering** with dendrogram visualisation

Evaluation metrics:

- Silhouette score
- Calinski-Harabasz index
- Davies-Bouldin index

Outputs cluster profiles with median/mean feature values.

### 5. Noise injection and denoising

**Notebook:** `ucl_data_noise.ipynb`

**Input:** `../data/processed/ucl_features_unscaled.parquet`  
**Output:**

- `../data/processed/ucl_features_noisy.parquet` (503 profiles with corrupted citations)
- `../data/processed/ucl_features_denoised.parquet` (503 profiles after outlier removal)

Simulates database corruption:

- Injects artificial noise into 5% of records (25 profiles)
- Corrupts `total_citations` by 10-20× multiplier
- Targets mid-range researchers (Q1-Q3 citation distribution)
- Recalculates derived features (`avg_citations_per_publication`, `annual_impact`)

Denoising strategy:

- **Isolation Forest** outlier detection (contamination=0.05)
- Uses 10 features for anomaly detection
- **Winsorization**: Caps outliers at upper IQR fence (Q3 + 1.5×IQR)

### 6. Supervised learning

**Notebook:** `ucl_data_supervised.ipynb`

**Input:** `../data/processed/ucl_final_clustered.parquet`

**Task:** Predict cluster assignments (0-4) using original features

**Model:** Random Forest Classifier

- Two-stage hyperparameter tuning (broad exploration → refined search)
- Balanced class weights for imbalanced clusters
- 128 estimators, max_features='sqrt'
- 80/20 train-test split with stratification

**Features:** 44 original unscaled features (excludes _index, name, email, profile_id, cluster)

Evaluation:

- Classification report
- Confusion matrix
- Accuracy and balanced accuracy

### 7. Noise robustness comparison

**Notebook:** `ucl_noise_clustering_comparison.ipynb`

Compares clustering quality across three datasets:

- Raw (original clean data)
- Noised (with 5% artificial corruption)
- Denoised (after Isolation Forest outlier removal)

Applies same preprocessing pipeline to all three:

- Log transformation
- StandardScaler
- PCA (95% variance)
- K-means clustering

Compares clustering metrics (silhouette, Calinski-Harabasz, Davies-Bouldin) and cluster stability using adjusted Rand
index.

## Feature schema

See [`features.md`](ucl_features.md) for complete documentation of all 43 engineered features, including feature names,
types, value ranges, transformations, and descriptions.

## Data requirements

### Input data

All notebooks expect `../data/raw/ucl_raw_aggregated.parquet` with schema:

**Profile metadata:**

- name, title, email, phone, department, university
- website, research_interests, bio, teaching_interests
- teaching_modules, teaching_module_count
- profile_id, profile_url
- _fetch_timestamp, _source

**Publication metrics (from aggregator):**

- publication_count, journal_article_count, preprint_count
- recent_publications, solo_publications, has_doi_count
- first_publication_year, latest_publication_year

**Citation metrics:**

- total_citations, avg_citations_per_publication, max_citations
- h_index, i10_index

**Collaboration metrics:**

- avg_coauthors, max_coauthors

### Output data

**`ucl_features_final.parquet`** (503 rows, 45 columns)

- 43 engineered features + _index
- Log-transformed and scaled

**`ucl_pca_features.parquet`** (503 rows, 14 columns)

- PC1 through PC13 (PCA components)
- _index (row identifier)

**`ucl_identity_mapping.parquet`** (503 rows)

- _index, name, email, profile_id
- title, website, research_interests, bio

**`ucl_features_noisy.parquet`** (503 rows, 45 columns)

- Unscaled features with 5% artificial noise

**`ucl_features_denoised.parquet`** (503 rows, 45 columns)

- Noisy data after Isolation Forest denoising

## Technical details

### Feature transformations

**Log transformations:**  
Applied to 19 skewed count features using log(x+1) to handle zeros: publication metrics, citation metrics, collaboration
metrics, temporal features, and derived productivity metrics.

**Normalisation:**  
StandardScaler (z-score) applied to log-transformed features.

**No scaling:**  
Binary indicators (14 research domains, 4 profile completeness, 2 engagement), ordinal features (seniority_level), and
ratio features (4 publication composition ratios) retained without scaling.

**Dimensionality reduction:**  
PCA reduces 44 features to 13 components capturing 95.3% variance (63.3% in PC1 alone).

**Data filtering:**  
Removes profiles with missing critical features, reducing dataset from 608 to 503 profiles.

### Missing data handling

- Empty names → Removed during aggregation
- Missing publications → Set to 0 (legitimate non-publishers)
- Missing profile fields → Transformed to binary flags (has_bio, has_website, has_research_interests, has_teaching_info)

## Common tasks

### Load engineered features

```python
import polars as pl

# Load final scaled features
df_features = pl.read_parquet('../data/processed/ucl_features_final.parquet')

# Load PCA features for clustering
df_pca = pl.read_parquet('../data/processed/ucl_pca_features.parquet')

# Load identity mapping
df_identity = pl.read_parquet('../data/processed/ucl_identity_mapping.parquet')

# Join if needed
df = df_pca.join(df_identity, on='_index')
```