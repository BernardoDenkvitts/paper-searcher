from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from arxiv_searcher import Paper


__vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

def preprocess_and_vectorize(papers: List[Paper]):
    """
    Extracts text from papers and vectorizes them using TF-IDF.
    Returns the vector matrix.
    """
    if not papers:
        return None
    
    documents = [f"{p.title} {p.abstract}" for p in papers]
    # Normalizing make its use cosine distance instead of euclidean distance
    return normalize(__vectorizer.fit_transform(documents), norm='l2')


def get_top_k_words(centroids, top_k=3):
    """
    Returns the top k words for each cluster based on centroid values.
    
    Args:
        centroids: Cluster centroids from KMeans (shape: n_clusters x n_features)
        top_k: Number of top words to extract per cluster
    
    Returns:
        Dictionary mapping cluster_id -> list of top k words
    """
    top_words = {}
    terms = __vectorizer.get_feature_names_out()
    
    for i in range(len(centroids)):
        # Get indices of top k features (words) with highest values in this centroid
        top_indices = centroids[i].argsort()[-top_k:][::-1]
        top_words[i] = [terms[idx] for idx in top_indices]
    
    return top_words


def get_2d_coordinates(X):
    """
    Reduces the dimensions of the vector matrix X to 2 components using PCA.
    """
    if X is None:
        return None
    
    # We need at least 2 samples/features for 2 components, but PCA handles it mostly. 
    # If samples < 2, it might error or return less components.
    if X.shape[0] < 2:
        return None

    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X.toarray())
