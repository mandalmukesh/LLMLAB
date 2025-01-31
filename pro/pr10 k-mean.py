from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

tfidf = TfidfVectorizer()
X = tfidf.fit_transform([s[0] for s in sentences])
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
print("Cluster Labels:", labels)