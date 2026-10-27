import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from time import perf_counter

SEED = 42
rng = np.random.default_rng(SEED)
specs = [(25, 38, 76, 130), (42, 85, 79, 120), (48, 62, 42, 130), (39, 89, 20, 120)]
parts = []
for age, income, spending, size in specs:
    parts.append(pd.DataFrame({
        'Age': rng.normal(age, 6, size).clip(18, 75),
        'Income': rng.normal(income, 11, size).clip(15, 150),
        'Spending': rng.normal(spending, 10, size).clip(1, 100),
    }))
customers = pd.concat(parts, ignore_index=True).sample(500, random_state=SEED).reset_index(drop=True)
scaled = StandardScaler().fit_transform(customers)
scores = []
for k in range(2, 9):
    model = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=SEED)
    labels = model.fit_predict(scaled)
    scores.append({'k': k, 'inertia': round(float(model.inertia_), 1), 'silhouette': round(float(silhouette_score(scaled, labels)), 3)})
chosen = max(scores, key=lambda x: x['silhouette'])['k']
model = KMeans(n_clusters=chosen, init='k-means++', n_init=30, random_state=SEED)
labels = model.fit_predict(scaled)
projection = PCA(n_components=2, random_state=SEED).fit_transform(scaled)
profile = customers.assign(Cluster=labels).groupby('Cluster').agg(Size=('Age','size'), Age=('Age','mean'), Income=('Income','mean'), Spending=('Spending','mean')).round(1).reset_index()

X, y = make_classification(n_samples=5000, n_features=300, n_informative=45, n_redundant=80, n_classes=3, class_sep=1.3, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
def evaluate(pipe):
    start = perf_counter(); pipe.fit(X_train,y_train); duration = perf_counter()-start
    return round(float(accuracy_score(y_test,pipe.predict(X_test))),3), round(duration,2), pipe
base = Pipeline([('scale',StandardScaler()),('model',RandomForestClassifier(n_estimators=180,n_jobs=-1,random_state=SEED))])
pca_pipe = Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=.95,svd_solver='full',random_state=SEED)),('model',RandomForestClassifier(n_estimators=180,n_jobs=-1,random_state=SEED))])
base_acc, base_time, _ = evaluate(base)
pca_acc, pca_time, fitted_pca = evaluate(pca_pipe)

digits = load_digits()
digits_30d = PCA(n_components=30, whiten=True, random_state=SEED).fit_transform(StandardScaler().fit_transform(digits.data))
emb = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=SEED).fit_transform(digits_30d)
sample_idx = np.linspace(0, len(labels)-1, 220, dtype=int)
digit_idx = np.linspace(0, len(digits.target)-1, 700, dtype=int)
result = {
    'scores': scores, 'chosen': chosen, 'profile': profile.to_dict(orient='records'),
    'scatter': [{'x':round(float(projection[i,0]),2),'y':round(float(projection[i,1]),2),'c':int(labels[i])} for i in sample_idx],
    'pca': {'base_acc':base_acc,'base_time':base_time,'pca_acc':pca_acc,'pca_time':pca_time,'components':int(fitted_pca.named_steps['pca'].n_components_)},
    'digits': [{'x':round(float(emb[i,0]),2),'y':round(float(emb[i,1]),2),'c':int(digits.target[i])} for i in digit_idx]
}
if '--summary' in __import__('sys').argv:
    result.pop('scatter')
    result.pop('digits')
print(json.dumps(result))
