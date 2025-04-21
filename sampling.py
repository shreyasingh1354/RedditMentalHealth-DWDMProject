import argparse
import gc
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def log_message(message, verbose=True):
    if verbose:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")


def sample_mental_health_data(
    df, 
    output_file=None,
    sampling_percentage=0.08, 
    max_per_subreddit=5000, 
    batch_size=1000,
    visualize=True,
    verbose=True
):
    start_time = time.time()
    log_message(f"Starting sampling process with target {sampling_percentage:.1%} of data", verbose)

    df = df.copy()

    df.drop(['title', 'selftext'], axis=1, inplace=True)
    df.rename(columns={"title_processed": "title", "selftext_processed": "text"}, inplace=True)

    df['text'] = df['text'].astype(str)
    if 'title' in df.columns:
        df['title'] = df['title'].astype(str)
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add year-month for temporal stratification
    df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')
    
    log_message(f"Dataset has {len(df)} posts across {df['subreddit'].nunique()} subreddits", verbose)
    log_message(f"Time range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}", verbose)
    
    # Initialize model for embeddings
    log_message("Loading sentence transformer model...", verbose)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    all_samples = []
    
    # Process each subreddit one by one
    subreddits = df['subreddit'].unique()
    for subreddit_idx, subreddit in enumerate(subreddits):
        log_message(f"\nProcessing subreddit {subreddit_idx+1}/{len(subreddits)}: {subreddit}", verbose)

        subreddit_df = df[df['subreddit'] == subreddit].copy()

        # IMP: Setting an Upper Bound for Memory Constraints
        MAX_ROWS = 450000
        if len(subreddit_df) > MAX_ROWS:
            log_message(f"WARNING: Limiting {subreddit} to {MAX_ROWS} rows (out of {len(subreddit_df)}) to prevent memory issues", verbose)
            subreddit_df = subreddit_df.sample(n=MAX_ROWS, random_state=42)

        target_size = min(max_per_subreddit, int(len(subreddit_df) * sampling_percentage))
        
        log_message(f"Target: {target_size} out of {len(subreddit_df)} posts", verbose)
        
        n_batches = (len(subreddit_df) + batch_size - 1) // batch_size

        log_message(f"Generating embeddings in {n_batches} batches...", verbose)
        all_embeddings = []
        
        for b in tqdm(range(n_batches), desc=f"Embedding {subreddit}", disable=not verbose):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, len(subreddit_df))
            
            batch_texts = subreddit_df['text'].iloc[start_idx:end_idx].tolist()
            
            # Generate embeddings for this batch
            batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        # Combine embeddings
        embeddings = np.vstack(all_embeddings)
        log_message(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}", verbose)
        
        # Cluster semantically
        n_clusters = min(20, max(5, len(subreddit_df) // 300))

        log_message(f"Clustering into {n_clusters} semantic groups using mini-batch...", verbose)
        
        mbk = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size*2,
            random_state=42,
            n_init=3,
            max_iter=100
        )
        clusters = mbk.fit_predict(embeddings)
        subreddit_df['cluster'] = clusters
        
        if verbose:
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            log_message(f"Cluster sizes: Min={cluster_counts.min()}, Max={cluster_counts.max()}, Mean={cluster_counts.mean():.1f}", verbose)
        
        # Free memory
        del embeddings
        del all_embeddings
        gc.collect()
        
        # Sample within each cluster, stratified by time
        log_message(f"Performing temporal stratification within clusters...", verbose)
        subreddit_samples = []
        
        for cluster_id in tqdm(range(n_clusters), desc=f"Sampling {subreddit}", disable=not verbose):
            cluster_df = subreddit_df[subreddit_df['cluster'] == cluster_id]
            if len(cluster_df) == 0:
                continue
                
            # Calculate proportional sample size
            cluster_proportion = len(cluster_df) / len(subreddit_df)
            cluster_target = max(1, int(target_size * cluster_proportion))
            
            # Sample temporally within cluster
            cluster_samples = []
            for month in cluster_df['year_month'].unique():
                month_df = cluster_df[cluster_df['year_month'] == month]
                month_proportion = len(month_df) / len(cluster_df)
                month_target = max(1, int(cluster_target * month_proportion))
                
                if month_target > 0:
                    samples = month_df.sample(
                        n=min(month_target, len(month_df)),
                        random_state=42
                    )
                    cluster_samples.append(samples)
            
            if cluster_samples:
                cluster_sample = pd.concat(cluster_samples)
                subreddit_samples.append(cluster_sample)
                if verbose:
                    tqdm.write(f"  Cluster {cluster_id}: sampled {len(cluster_sample)} posts")
        
        if subreddit_samples:
            subreddit_sample = pd.concat(subreddit_samples)
            all_samples.append(subreddit_sample)
            log_message(f"Sampled {len(subreddit_sample)} posts from {subreddit} ({len(subreddit_sample)/len(subreddit_df):.1%})", verbose)
        else:
            log_message(f"Warning: No samples generated for {subreddit}!", verbose)
        
        # Free more memory
        del subreddit_df
        gc.collect()
    
    # Combine all samples
    if not all_samples:
        log_message("Warning: No samples were generated!", verbose)
        return pd.DataFrame()
        
    result = pd.concat(all_samples)
    
    # Remove cluster column
    if 'cluster' in result.columns:
        result = result.drop(columns=['cluster'])
    
    # Total time taken
    elapsed_time = time.time() - start_time
    log_message(f"Sampling completed in {elapsed_time/60:.1f} minutes", verbose)
    log_message(f"Final sample: {len(result)} posts ({len(result)/len(df):.1%} of original)", verbose)
    
    # Save to file if specified
    if output_file:
        log_message(f"Saving sampled dataset to {output_file}...", verbose)
        result.to_csv(output_file, index=False)
    
    if visualize:
        log_message("Generating visualizations...", verbose)
        visualize_sampling_results(df, result)
    
    return result


def visualize_sampling_results(original_df, sampled_df):
    # Set up plotting
    plt.figure(figsize=(16, 12))
    
    # 1. Subreddit distribution
    plt.subplot(1, 2, 1)
    comparison = pd.DataFrame({
        'Original': original_df['subreddit'].value_counts(normalize=True),
        'Sampled': sampled_df['subreddit'].value_counts(normalize=True)
    })
    comparison.plot(kind='bar', ax=plt.gca())
    plt.title('Subreddit Distribution: Original vs Sampled')
    plt.xlabel('Subreddit')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    
    # 2. Temporal distribution
    plt.subplot(1, 2, 2)
    original_monthly = original_df.groupby('year_month').size()
    sampled_monthly = sampled_df.groupby('year_month').size()

    # Normalize for comparison
    original_monthly_norm = original_monthly / original_monthly.sum()
    sampled_monthly_norm = sampled_monthly / sampled_monthly.sum()

    pd.DataFrame({
        'Original': original_monthly_norm,
        'Sampled': sampled_monthly_norm
    }).plot(kind='line', ax=plt.gca())
    plt.title('Temporal Distribution: Original vs Sampled')
    plt.xlabel('Month')
    plt.ylabel('Proportion of Posts')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('sampling_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualizations saved to 'sampling_results.png'")


def main():
    parser = argparse.ArgumentParser(description='Sample mental health Reddit data using semantic and temporal clustering.')
    parser.add_argument('input_file', help='Input CSV file containing the Reddit data')
    parser.add_argument('output_file', help='Output CSV file to save the sampled data')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return 1
    
    print(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Run the sampling
    sampled_df = sample_mental_health_data(
        df=df,
        output_file=args.output_file
    )
    
    print(f"Process complete. Sampled {len(sampled_df)} posts from {len(df)} original posts.")
    return 0


if __name__ == '__main__':
    main()