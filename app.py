import streamlit as st
import pandas as pd
import torch
import os

from src.data_loader import load_data
from utils.visualizations import visualize_data
from src.gan_model import Generator, Discriminator
from src.generate import generate_synthetic_data, decode_synthetic_data

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
    uploaded_file = st.file_uploader("📂 Upload `mock_data.csv`", type=["csv"])

    if uploaded_file:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("❌ Uploaded file is empty.")
                return

            st.subheader("📊 Preview of Uploaded Data")
            st.dataframe(df.head())

            uploaded_file.seek(0)
            real_data, label_encoders = load_data(uploaded_file)
            st.session_state["real_data"] = real_data
            st.session_state["label_encoders"] = label_encoders

            st.subheader("📈 Data Visualizations")
            visualize_data(real_data)

            if st.button("🚀 Generate Synthetic Data"):
                noise_dim = 16
                generator = Generator(input_dim=noise_dim, output_dim=real_data.shape[1])
                discriminator = Discriminator(input_dim=real_data.shape[1])

                st.subheader("🛠️ Training GAN...")
                progress_bar = st.progress(0, text="Starting training...")

                # Manual GAN training loop
                num_epochs = 200
                batch_size = 32
                learning_rate = 0.0002

                optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
                optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
                criterion = torch.nn.BCELoss()

                real_tensor = torch.tensor(real_data.values, dtype=torch.float32)
                dataloader = torch.utils.data.DataLoader(real_tensor, batch_size=batch_size, shuffle=True)

                for epoch in range(1, num_epochs + 1):
                    for real_batch in dataloader:
                        batch_size_curr = real_batch.size(0)
                        real_labels = torch.ones((batch_size_curr, 1))
                        fake_labels = torch.zeros((batch_size_curr, 1))

                        noise = torch.randn(batch_size_curr, noise_dim)
                        fake_data = generator(noise)

                        d_real = discriminator(real_batch)
                        d_fake = discriminator(fake_data.detach())

                        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
                        optimizer_D.zero_grad()
                        d_loss.backward()
                        optimizer_D.step()

                        noise = torch.randn(batch_size_curr, noise_dim)
                        fake_data = generator(noise)
                        g_loss = criterion(discriminator(fake_data), real_labels)
                        optimizer_G.zero_grad()
                        g_loss.backward()
                        optimizer_G.step()

                    percent_complete = int((epoch / num_epochs) * 100)
                    progress_bar.progress(percent_complete, text=f"Training GAN... {percent_complete}%")

                progress_bar.empty()
                st.success("✅ GAN training completed!")

                # Generate synthetic data
                synthetic_df = generate_synthetic_data(
                    generator,
                    num_samples=len(real_data),
                    noise_dim=noise_dim,
                    column_names=real_data.columns.tolist()
                )
                synthetic_df = decode_synthetic_data(synthetic_df, label_encoders)

                st.subheader("🧬 Synthetic Data Preview")
                st.dataframe(synthetic_df.head())

                csv = synthetic_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Synthetic CSV", csv, file_name="synthetic_output.csv", mime='text/csv')

                # 🔄 Reset
                if st.button("🔁 Reset"):
                    st.session_state.clear()
                    st.experimental_rerun()

        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded CSV file is empty or invalid.")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")

    # Footer
    st.markdown(
        """
        <hr style="margin-top: 50px;"/>
        <div style="text-align: center; font-size: 0.9rem; color: gray;">
            © 2025 SynGenie • Built with Streamlit, PyTorch, and 💡
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
