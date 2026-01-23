from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from arxiv_searcher import Paper


def preprocess_and_vectorize(papers: List[Paper]):
    """
    Extracts text from papers and vectorizes them using TF-IDF.
    Returns the vector matrix.
    """
    if not papers:
        return None
    
    documents = [f"{p.title} {p.abstract}" for p in papers]
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    return vectorizer.fit_transform(documents)


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
