import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
import time  # Added missing import
from scipy.stats import wasserstein_distance
from src.data_loader import load_data
from utils.visualizations import visualize_data
from src.gan_model import Generator, Discriminator
from src.generate import generate_synthetic_data, decode_synthetic_data

def calculate_similarity(real_data, synthetic_data):
    """Calculate quality score between real and synthetic data (0-1 scale)"""
    scores = []
    for col in real_data.select_dtypes(include=np.number).columns:
        if col in synthetic_data.columns:
            try:
                real_vals = real_data[col].dropna().values
                synth_vals = synthetic_data[col].dropna().values
                if len(real_vals) > 1 and len(synth_vals) > 1:
                    dist = wasserstein_distance(real_vals, synth_vals)
                    data_range = max(real_vals.max() - real_vals.min(), 1e-8)
                    scores.append(1 - min(dist/data_range, 1))
            except:
                continue
    return np.mean(scores) if scores else 0.0

def clean_dataframe(df, label_encoders):
    """Enhanced cleaning with proper type handling for boolean columns"""
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for col in df.columns:
        if not df[col].isnull().all():
            if col in label_encoders:
                # Handle categorical columns
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(mode_val)
                df[col] = df[col].round().astype(int)
                df[col] = np.clip(df[col], 0, len(label_encoders[col].classes_)-1)
            else:
                # Handle numerical columns - ensure proper numeric type
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Only calculate statistics if we have valid numbers
                if not df[col].isnull().all():
                    median = df[col].median()
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    if iqr > 0:
                        # Use numpy clip instead of pandas clip for better type handling
                        lower_bound = q1 - 1.5*iqr
                        upper_bound = q3 + 1.5*iqr
                        df[col] = np.clip(df[col], lower_bound, upper_bound)
                    
                    df[col] = df[col].fillna(median)
        else:
            df[col] = 0 if col in label_encoders else 0.0
    
    return df

def generate_enhanced_data(generator, num_samples, noise_dim, columns, data_min, data_max, label_encoders):
    """Generate synthetic data with proper boolean handling"""
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim)
        synthetic = torch.sigmoid(generator(noise))
        
        # Convert to numpy and handle boolean types
        synthetic_np = synthetic.cpu().numpy()
        
        # Convert any boolean arrays to float32 to allow arithmetic operations
        if synthetic_np.dtype == bool:
            synthetic_np = synthetic_np.astype(np.float32)
        
        # Perform denormalization on properly typed data
        synthetic_np = synthetic_np * (data_max.numpy() - data_min.numpy()) + data_min.numpy()
        
        df = pd.DataFrame(synthetic_np, columns=columns)
        df = clean_dataframe(df, label_encoders)
        return df

def display_results(real_data, synthetic_df):
    """Enhanced display with total rows and expanded preview options"""
    st.subheader("📊 Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Data Rows", len(real_data))
    with col2:
        st.metric("Synthetic Data Rows", len(synthetic_df))
    with col3:
        st.metric("Data Quality Score", f"{calculate_similarity(real_data, synthetic_df):.1%}")
    
    st.subheader("🧬 Synthetic Data Exploration")
    
    # Preview options
    preview_option = st.radio("Preview Type:", 
                             ["First 10 Rows", "Last 10 Rows", "Random Sample", "Full Data"], 
                             horizontal=True)
    
    if preview_option == "First 10 Rows":
        st.dataframe(synthetic_df.head(10))
    elif preview_option == "Last 10 Rows":
        st.dataframe(synthetic_df.tail(10))
    elif preview_option == "Random Sample":
        st.dataframe(synthetic_df.sample(10))
    else:
        # Full data view with pagination
        page_size = st.select_slider("Rows per page:", 
                                   options=[10, 25, 50, 100, 250, 500],
                                   value=25)
        total_pages = (len(synthetic_df)) // page_size + 1
        page_number = st.number_input("Page:", 
                                    min_value=1, 
                                    max_value=total_pages, 
                                    value=1)
        
        start_idx = (page_number-1)*page_size
        end_idx = start_idx + page_size
        st.dataframe(synthetic_df.iloc[start_idx:end_idx])
    
    # Download options
    st.subheader("📥 Download Options")
    csv = synthetic_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Full Dataset as CSV", 
        csv, 
        file_name="synthetic_data.csv", 
        mime='text/csv'
    )

def estimate_generation_time(file_size_mb, num_rows, num_cols):
    """Estimate generation time based on file size and dimensions"""
    # Base time in seconds for small files (<1MB)
    base_time = 15  
    
    # Adjust for file size (linear scaling)
    size_factor = max(1, file_size_mb * 0.2)
    
    # Adjust for data complexity (rows * columns)
    complexity_factor = max(1, (num_rows * num_cols) / 10000)
    
    # Estimated time in minutes
    est_minutes = (base_time * size_factor * complexity_factor) / 60
    
    # Cap at 30 minutes for very large files
    return min(30, max(1, round(est_minutes)))

def main():
    st.set_page_config(page_title="Synthetic Data Generator", layout="wide")
    st.title("🗂️ SynGenie — Synthetic Data Generator")

    # ℹ️ About Section
    with st.expander("ℹ️ About SynGenie"):
        st.markdown("""
        **SynGenie** is an AI-powered synthetic data generation tool powered by **Generative Adversarial Networks (GANs)**.

        It helps you create realistic yet anonymous datasets based on your original data — ideal for testing, privacy-preserving data sharing, or machine learning experimentation.

        ### 🧠 How it Works:
        1. **Upload** a `.csv` file with your real tabular data.
        2. SynGenie **analyzes and encodes** your data, including categorical columns.
        3. It **trains a GAN model** to learn the distribution and patterns.
        4. Once trained, it **generates synthetic data** that mirrors your input — but without revealing actual values.

        **No data leaves your machine**. Everything runs locally and securely.
        """)

    # 📥 Download Sample CSV
    st.sidebar.subheader("📁 Sample Data")
    sample_path = "sample/mock_data.csv"
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            sample_bytes = f.read()
        st.sidebar.download_button(
            label="⬇️ Download mock_data.csv",
            data=sample_bytes,
            file_name="mock_data.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.warning("Sample file not found.")

    # File Uploader
    uploaded_file = st.file_uploader("📂 Upload your data file", type=["csv"])

    if uploaded_file:
        try:
            # Load and prepare data
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("❌ Uploaded file is empty.")
                return
            
            # Calculate file size in MB
            file_size_bytes = len(uploaded_file.getvalue())
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Calculate estimated time
            est_time = estimate_generation_time(
                file_size_mb=file_size_mb,
                num_rows=len(df),
                num_cols=len(df.columns)
            )
            
            st.subheader("📊 Original Data Overview")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Rows", len(df))
            with cols[1]:
                st.metric("Total Columns", len(df.columns))
            with cols[2]:
                st.metric("Estimated Generation Time", f"{est_time:.0f} minutes")
            
            # Enhanced original data preview
            st.subheader("🔍 Data Exploration")
            preview_type = st.radio("View:", 
                                  ["Quick Preview", "Full Data"], 
                                  horizontal=True,
                                  key="original_preview")
            
            if preview_type == "Quick Preview":
                tab1, tab2, tab3 = st.tabs(["First 10 Rows", "Last 10 Rows", "Random Sample"])
                with tab1:
                    st.dataframe(df.head(10))
                with tab2:
                    st.dataframe(df.tail(10))
                with tab3:
                    st.dataframe(df.sample(10))
            else:
                # Full data view with pagination
                page_size = st.select_slider("Rows per page:", 
                                           options=[10, 25, 50, 100, 250, 500],
                                           value=25,
                                           key="original_page_size")
                total_pages = (len(df) // page_size + 1)
                page_number = st.number_input("Page:", 
                                            min_value=1, 
                                            max_value=total_pages, 
                                            value=1,
                                            key="original_page_num")
                
                start_idx = (page_number-1)*page_size
                end_idx = start_idx + page_size
                st.dataframe(df.iloc[start_idx:end_idx])

            real_data, label_encoders = load_data(uploaded_file)
            real_data = clean_dataframe(real_data, label_encoders)
            
            st.session_state["real_data"] = real_data
            st.session_state["label_encoders"] = label_encoders

            st.subheader("📈 Data Visualizations")
            visualize_data(real_data)

            if st.button("🚀 Generate Synthetic Data"):
                # Show time estimation
                with st.expander("ℹ️ Generation Details"):
                    st.write(f"File Size: {file_size_mb:.2f} MB")
                    st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
                    st.write(f"Estimated time: {est_time} minutes")
                    st.write("Actual time may vary based on your system hardware")
                
                start_time = time.time()

                # Store noise_dim in session state
                st.session_state['noise_dim'] = 16
                generator = Generator(input_dim=st.session_state['noise_dim'], 
                                    output_dim=real_data.shape[1])
                discriminator = Discriminator(input_dim=real_data.shape[1])

                st.subheader("🛠️ Training GAN...")
                progress_bar = st.progress(0, text="Starting training...")

                # Training parameters
                num_epochs = 200
                batch_size = 32
                learning_rate = 0.0002

                # Initialize optimizers and loss
                optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
                optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
                criterion = torch.nn.BCEWithLogitsLoss()

                # Prepare and normalize data
                real_tensor = torch.tensor(real_data.values, dtype=torch.float32)
                data_min = real_tensor.min(dim=0).values
                data_max = real_tensor.max(dim=0).values
                real_tensor = (real_tensor - data_min) / (data_max - data_min + 1e-8)
                real_tensor = torch.clamp(real_tensor, 0.0, 1.0)
                
                # Store normalization parameters
                st.session_state['data_min'] = data_min
                st.session_state['data_max'] = data_max
                
                # Training loop
                for epoch in range(1, num_epochs + 1):
                    for real_batch in torch.utils.data.DataLoader(real_tensor, batch_size=batch_size, shuffle=True):
                        # Train discriminator
                        optimizer_D.zero_grad()
                        real_labels = torch.FloatTensor(real_batch.size(0), 1).uniform_(0.9, 1.0)
                        d_real = discriminator(real_batch)
                        d_loss_real = criterion(d_real, real_labels)
                        
                        noise = torch.randn(real_batch.size(0), st.session_state['noise_dim'])
                        fake_data = torch.sigmoid(generator(noise))
                        fake_labels = torch.FloatTensor(real_batch.size(0), 1).uniform_(0.0, 0.1)
                        d_fake = discriminator(fake_data.detach())
                        d_loss_fake = criterion(d_fake, fake_labels)
                        
                        d_loss = d_loss_real + d_loss_fake
                        d_loss.backward()
                        optimizer_D.step()
                        
                        # Train generator
                        optimizer_G.zero_grad()
                        noise = torch.randn(real_batch.size(0), st.session_state['noise_dim'])
                        fake_data = torch.sigmoid(generator(noise))
                        g_loss = criterion(discriminator(fake_data), real_labels)
                        g_loss.backward()
                        optimizer_G.step()

                    progress_bar.progress(int((epoch / num_epochs) * 100), 
                                       text=f"Training GAN... {epoch}/{num_epochs} epochs")

                # Store trained models
                st.session_state['generator'] = generator
                st.session_state['discriminator'] = discriminator
                
                # Generate initial synthetic data
                synthetic_df = generate_enhanced_data(
                    generator, 
                    len(real_data), 
                    st.session_state['noise_dim'],
                    real_data.columns, 
                    data_min, 
                    data_max, 
                    label_encoders
                )
                synthetic_df = decode_synthetic_data(synthetic_df, label_encoders)
                
                # Store for regeneration
                st.session_state['synthetic_df'] = synthetic_df
                
                # Display results
                end_time = time.time()
                actual_minutes = (end_time - start_time) / 60
                st.success(f"✅ GAN training completed in {actual_minutes:.1f} minutes!")
                display_results(real_data, synthetic_df)

            # Regenerate with new noise
            if st.button("🔄 Regenerate with New Noise", 
                         disabled=not ('generator' in st.session_state)):
                if 'generator' not in st.session_state:
                    st.error("Please generate data first")
                    return
                
                with st.spinner("Generating fresh synthetic data..."):
                    synthetic_df = generate_enhanced_data(
                        st.session_state['generator'], 
                        len(st.session_state["real_data"]), 
                        st.session_state['noise_dim'],
                        st.session_state["real_data"].columns,
                        st.session_state['data_min'], 
                        st.session_state['data_max'],
                        st.session_state["label_encoders"]
                    )
                    synthetic_df = decode_synthetic_data(synthetic_df, st.session_state["label_encoders"])
                    st.session_state['synthetic_df'] = synthetic_df
                
                st.success("🔄 Synthetic data regenerated with new random noise!")
                display_results(st.session_state["real_data"], synthetic_df)

            # Full reset
            if st.button("🔁 Full Reset"):
                st.session_state.clear()
                st.rerun()

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

    # Footer
    st.markdown("""<hr/><div style="text-align: center; font-size: 0.9rem; color: gray;">
                © 2025 SynGenie • Built with Streamlit, PyTorch, and 💡</div>""", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()