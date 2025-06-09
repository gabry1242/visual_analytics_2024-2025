import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os

def precompute_tsne_data():
    """
    Precompute TSNE embeddings and cluster assignments for different cluster counts
    and save them to disk for fast loading in the main app.
    """
    
    # Load and prepare data (same as your main app)
    df = pd.read_csv("merged_with_tags.csv")
    df = df.dropna(subset=["budget", "revenue", "release_year", "title_y", "vote_count", 'genres_y'])
    df = df.drop_duplicates(subset=['title_y'])
    df["release_year"] = df["release_year"].astype(int)
    
    # Extra metrics
    df["profit"] = df["revenue"] - df["budget"]
    df["profit_margin"] = (df["profit"] / df["budget"]).replace([np.inf, -np.inf], np.nan)
    df["roi"] = df["profit_margin"] * 100
    df["primary_genre"] = df["genres_y"].str.split("-").str[0]
    
    # Define features used for clustering (same as your main app)
    numeric_features = ["budget", "revenue", "vote_average", "vote_count", "runtime", "profit", "roi"]
    
    print("Starting TSNE precomputation...")
    print(f"Dataset shape: {df.shape}")
    
    # Prepare data for TSNE
    df_clean = df.dropna(subset=numeric_features + ["primary_genre"])
    print(f"Clean dataset shape: {df_clean.shape}")
    
    X = df_clean[numeric_features]
    X_scaled = StandardScaler().fit_transform(X)
    
    # Compute TSNE once (this is the expensive operation)
    print("Computing TSNE embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    tsne_components = tsne.fit_transform(X_scaled)
    
    # Add TSNE coordinates to dataframe
    df_clean = df_clean.copy()
    df_clean["tsne_dim1"] = tsne_components[:, 0]
    df_clean["tsne_dim2"] = tsne_components[:, 1]
    
    # Precompute clusters for different k values (2 to 10)
    print("Computing cluster assignments...")
    cluster_results = {}
    
    for k in range(2, 11):
        print(f"Computing clusters for k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        cluster_results[k] = cluster_labels.astype(str)
    
    # Create the final data structure to save
    tsne_data = {
        'dataframe': df_clean,  # Contains all original data + tsne_dim1, tsne_dim2
        'clusters': cluster_results,  # Dictionary mapping k -> cluster labels
        'numeric_features': numeric_features,
        'scaled_features': X_scaled  # In case you need it later
    }
    
    # Save to pickle file
    print("Saving precomputed data...")
    with open('tsne_precomputed.pkl', 'wb') as f:
        pickle.dump(tsne_data, f)
    
    print("TSNE precomputation complete!")
    print(f"Saved data for {len(df_clean)} movies")
    print(f"Cluster configurations: k={list(cluster_results.keys())}")
    print("File saved as: tsne_precomputed.pkl")
    
    return tsne_data

def load_tsne_data():
    """
    Load precomputed TSNE data from disk.
    Returns None if file doesn't exist.
    """
    if os.path.exists('tsne_precomputed.pkl'):
        with open('tsne_precomputed.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        print("Precomputed TSNE file not found. Run precompute_tsne_data() first.")
        return None

if __name__ == "__main__":
    # Run the precomputation
    tsne_data = precompute_tsne_data()
    
    # Test loading
    print("\nTesting data loading...")
    loaded_data = load_tsne_data()
    if loaded_data:
        print("✓ Data loaded successfully")
        print(f"✓ DataFrame shape: {loaded_data['dataframe'].shape}")
        print(f"✓ Available cluster configurations: {list(loaded_data['clusters'].keys())}")
        print(f"✓ TSNE dimensions available: tsne_dim1, tsne_dim2")
    else:
        print("✗ Failed to load data")