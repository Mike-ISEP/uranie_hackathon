# 🎈 URANIE DEFENSE

The Uranie Defense System is a human-in-the-loop anomaly detection web interface built with Streamlit. It leverages an Ensemble Denoising Autoencoder (EDAE) to detect anomalies in high-dimensional data and incorporates active learning for continuous improvement via user labeling.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

# 📁 Project Structure

├── streamlit_app.py          # Main Streamlit web app
├── workshop_s1.txt           # Input dataset (tab-delimited)
├── AEmodel.py                # Ensemble Denoising Autoencoder class
├── sorting_function.py       # Uncertainty sorting logic
├── t_update_function.py      # Adaptive threshold update logic
└── README.md                 # You're reading it!

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
