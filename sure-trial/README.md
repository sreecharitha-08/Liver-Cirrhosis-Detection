Streamlit app: Liver Cirrhosis Stage Prediction

Quick summary
- This repository contains a Streamlit app (`app.py`) that predicts cirrhosis stage using a trained model (`cirrhosis_model.pkl`) and a scaler (`scaler.pkl`).

Common deployment errors on Streamlit Cloud and how to fix them
1. Missing model/scaler files
   - Streamlit Cloud runs from your git repo. Ensure `cirrhosis_model.pkl` and `scaler.pkl` are committed to the repository root (the same folder as `app.py`).
   - These files are binary. If they are large (>100 MB) consider using Git LFS or hosting them elsewhere and downloading at runtime.

2. Absolute Windows paths in code
   - Avoid absolute paths like `C:\Users\...`. The app now uses relative paths and looks for the files next to `app.py`.

3. Missing dependencies
   - Add a `requirements.txt` (included) and ensure Streamlit Cloud installs packages from it. If you add other libraries, update `requirements.txt`.

How to generate the model files locally
1. Ensure you have the dataset available locally and update `generate_model_files.py` paths if necessary.
2. Run `python generate_model_files.py` locally. This will produce `cirrhosis_model.pkl` and `scaler.pkl` in the working directory.
3. Commit and push those files to your repo.

Deploying to Streamlit Cloud (recommended flow)
1. Push your repository to GitHub (include `app.py`, `requirements.txt`, `cirrhosis_model.pkl`, `scaler.pkl`).
2. Go to https://share.streamlit.io and connect your GitHub account.
3. Create a new app and point it to this repo, selecting `app.py` as the main file.
4. Check the app logs on Streamlit Cloud if the deployment fails. Logs show Python exceptions and missing-file errors.

If your files are too large for git
- Use Git LFS or host the pickle files on a small object store and download them in `app.py` at runtime. If you choose to download at runtime, make sure to cache them locally (e.g., with Streamlit's `st.cache_data`) so you don't fetch every request.

Debugging tips
- Open app logs on Streamlit Cloud — they often reveal missing dependencies or FileNotFoundError for the model files.
- If you see model feature errors, inspect the model's `feature_names_in_` (the app will print them when available).

If you'd like, I can:
- Add a small script to download model files at runtime from a given URL and cache them.
- Help you create Git LFS pointers or a `setup` script to build the model within CI.

Contact
- If you share the Streamlit Cloud logs (or the exact error text), I can diagnose further and provide a precise fix.