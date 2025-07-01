# 🗂️ SynGenie — Synthetic Data Generator

SynGenie is a web-based tool built with **Streamlit** and **PyTorch** that uses **Generative Adversarial Networks (GANs)** to generate realistic synthetic tabular data based on your real CSV dataset. This is useful for data privacy, testing, and simulation purposes.

---

## 🚀 Features

- Upload your own `.csv` file
- Visualize your real data
- Train a GAN model in-browser with a progress bar
- Preview and download the generated synthetic data
- Clean, user-friendly dashboard interface
- Respects data privacy (no server-side upload)

---

## 📦 Installation

### 🔧 Prerequisites

- Python 3.8–3.12
- `pip` package manager

### 🛠️ Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/syngenie.git
cd syngenie

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

If you don't have a requirements.txt yet, here's a sample you can create:

text
Copy
Edit
streamlit
pandas
torch
seaborn
matplotlib
scikit-learn

▶️ Running the App
To launch the dashboard in your browser:

bash
Copy
Edit
streamlit run app.py
Then open your browser to http://localhost:8501 if it doesn't open automatically.

📁 Project Structure
graphql
Copy
Edit
syngenie/
├── app.py                      # Main Streamlit dashboard
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py          # Data preprocessing and encoding
│   ├── train.py                # GAN training loop (optional)
│   ├── gan_model.py            # Generator & Discriminator classes
│   └── generate.py             # Synthetic data generation + decoding
├── utils/
│   └── visualizations.py       # Real data visualizations
📄 Example CSV Format
Make sure your input CSV:

Is not empty

Has column headers

Contains only numeric or categorical columns

cs
Copy
Edit
Age,Gender,Salary
23,Male,50000
35,Female,60000
...
🛡️ Disclaimer
This tool is for educational and experimental purposes. Always verify the validity and privacy safety of synthetic data before using it in production.

👨‍💻 Author
Developed by Keith Andrew Relles, Shivendra Rajput — Contributions welcome!
