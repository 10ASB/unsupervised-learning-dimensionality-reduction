# Unsupervised Learning & Dimensionality Reduction

This project contains the Day 9 practical script for customer segmentation, PCA-based dimensionality reduction, and t-SNE visualization.

## Actual practicals in `day9.py`

### 1. Customer segmentation with K-Means
- Generates synthetic customer data with `Age`, `Income`, and `Spending`.
- Standardizes the features.
- Tests K-Means cluster counts from 2 through 8.
- Records inertia and silhouette score for each candidate.
- Chooses the cluster count with the highest silhouette score.
- Profiles clusters by size, average age, income, and spending.
- Uses PCA to project the customer data into two dimensions.

### 2. PCA before classification
- Creates a synthetic three-class dataset with 5,000 samples and 300 features.
- Uses a stratified train/test split.
- Compares a scaled Random Forest pipeline with a scaled PCA + Random Forest pipeline.
- PCA retains enough components to explain 95% of variance.
- Records test accuracy, training time, and retained component count.

### 3. PCA + t-SNE visualization on digits
- Loads scikit-learn's handwritten digits dataset.
- Scales the pixel features and reduces them to 30 dimensions using PCA.
- Uses t-SNE to create a two-dimensional embedding.
- Emits sampled embedding coordinates and digit labels.

## Files
- `day9.py` — the practical script supplied for this project.
- `README.md` — project overview and run instructions.
- `day9_lab_guide.md` — tasks and reporting checklist.

## Requirements
```bash
pip install numpy pandas scikit-learn
```

## Run
```bash
python day9.py
```
The script prints JSON containing clustering scores, cluster profiles, PCA comparison metrics, and sampled coordinates.

For a compact summary without sampled customer and digits coordinates:
```bash
python day9.py --summary
```

## Interpretation notes
- K-Means uses standardized features because age, income, and spending have different scales.
- The silhouette score selects a candidate cluster count; it does not prove that this is the only meaningful grouping.
- The PCA comparison reports accuracy and elapsed training time; values depend on the runtime environment.
- The script emits coordinates rather than creating image files itself.
