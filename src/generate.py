import torch
import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress tracking

def generate_synthetic_data(generator, num_samples=100, noise_dim=16, column_names=None, 
                          batch_size=1024, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate synthetic data using the trained generator with optimized batch processing.
    
    Args:
        generator (Generator): Trained GAN generator
        num_samples (int): Number of samples to generate
        noise_dim (int): Dimension of noise input
        column_names (list): Column names for the output DataFrame
        batch_size (int): Number of samples to generate per batch
        device (str): Device to use for generation ('cuda' or 'cpu')
        
    Returns:
        pd.DataFrame: Synthetic data with specified column names
    """
    generator.to(device)
    generator.eval()
    
    synthetic_data = []
    remaining_samples = num_samples
    
    with torch.no_grad():
        # Process in batches for memory efficiency
        for _ in tqdm(range(0, num_samples, batch_size), desc="Generating synthetic data"):
            current_batch_size = min(batch_size, remaining_samples)
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            batch = generator(noise)
            
            # Convert to numpy safely, handling any boolean types
            batch_np = batch.cpu().numpy()
            if batch_np.dtype == bool:
                batch_np = batch_np.astype(np.float32)
            
            synthetic_data.append(batch_np)
            remaining_samples -= current_batch_size
    
    # Concatenate all batches and create DataFrame
    synthetic_array = np.concatenate(synthetic_data, axis=0)
    return pd.DataFrame(synthetic_array[:num_samples], columns=column_names)

def decode_synthetic_data(df, label_encoders, batch_size=10000):
    """
    Decode synthetic data with optimized batch processing and error handling.
    
    Args:
        df (pd.DataFrame): Synthetic data to decode
        label_encoders (dict): Dictionary of label encoders for categorical columns
        batch_size (int): Number of rows to process at once
        
    Returns:
        pd.DataFrame: Decoded synthetic data
    """
    decoded_df = df.copy()
    
    for col, le in label_encoders.items():
        if col in decoded_df.columns:
            try:
                # Convert to numeric first to avoid boolean issues
                decoded_df[col] = pd.to_numeric(decoded_df[col], errors='coerce')
                
                # Process in batches for memory efficiency
                for i in range(0, len(decoded_df), batch_size):
                    batch = decoded_df[col].iloc[i:i+batch_size]
                    
                    # Convert to integers safely, handling any boolean types
                    batch = batch.round().astype(int)
                    
                    # Clip to valid range
                    max_class = len(le.classes_) - 1
                    batch = np.clip(batch, 0, max_class)
                    
                    # Inverse transform
                    decoded_df[col].iloc[i:i+batch_size] = le.inverse_transform(batch)
                    
            except Exception as e:
                print(f"Decoding error on column '{col}': {e}")
                decoded_df[col] = "Unknown"
    
    return decoded_df