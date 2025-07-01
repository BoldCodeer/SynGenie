import torch
import pandas as pd

def generate_synthetic_data(generator, num_samples=100, noise_dim=16, column_names=None):
    """
    Generate synthetic data using the trained generator.

    Args:
        generator (Generator): Trained GAN generator.
        num_samples (int): Number of samples to generate.
        noise_dim (int): Dimension of noise input.
        column_names (list, optional): Column names to assign to the synthetic DataFrame.

    Returns:
        pd.DataFrame: Synthetic data.
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim)
        synthetic = generator(noise).numpy()

    return pd.DataFrame(synthetic, columns=column_names)

def decode_synthetic_data(df, label_encoders):
    """
    Decode categorical columns in synthetic data using stored label encoders.

    Args:
        df (pd.DataFrame): Synthetic data with numeric-encoded categorical columns.
        label_encoders (dict): Dictionary of LabelEncoders used during training.

    Returns:
        pd.DataFrame: Decoded synthetic data.
    """
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.inverse_transform(
                    df[col].round().astype(int).clip(0, len(le.classes_) - 1)
                )
            except Exception as e:
                print(f"Warning: Could not decode column '{col}': {e}")
    return df
