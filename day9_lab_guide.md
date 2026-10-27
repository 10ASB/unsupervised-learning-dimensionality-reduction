# Day 9 Lab Guide — Unsupervised Learning & Dimensionality Reduction

This guide follows the tasks implemented in the supplied `day9.py` script.

## Task 1 — K-Means customer segmentation
1. Generate synthetic customer data for age, income, and spending.
2. Standardize the three features.
3. Fit K-Means for k values from 2 through 8.
4. Record inertia and silhouette score for each k.
5. Select the k with the highest silhouette score and profile the clusters.
6. Explain what each cluster's size, average age, income, and spending suggest.

## Task 2 — PCA and classification
1. Create the three-class, 300-feature classification dataset.
2. Compare the scaled Random Forest baseline against the scaled PCA + Random Forest pipeline.
3. Record test accuracy and fit time for both.
4. Report how many PCA components were retained to preserve 95% variance.
5. Discuss the trade-off between dimensionality reduction, model performance, and training time.

## Task 3 — Visualizing handwritten digits
1. Load the digits dataset.
2. Standardize pixel features and reduce them to 30 dimensions using PCA.
3. Apply t-SNE to produce a 2D embedding.
4. Use digit labels to color the embedding in a scatter plot.
5. Explain that t-SNE is primarily a visualization method and that distances in a 2D embedding should be interpreted carefully.

## Run
```bash
pip install numpy pandas scikit-learn
python day9.py
```
Use `python day9.py --summary` to omit sampled coordinates from the JSON output.

## Reporting checklist
- [ ] Show inertia and silhouette scores for k = 2 through 8.
- [ ] State the k selected by the script.
- [ ] Summarize customer cluster profiles.
- [ ] Compare baseline and PCA pipeline test accuracy and training time.
- [ ] Report the number of PCA components retained.
- [ ] Explain the PCA-to-t-SNE workflow for digit visualization.
- [ ] Separate observed run results from general interpretation.
