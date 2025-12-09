# Skripsi - Analisis Emosi

Small Streamlit project for emotion analysis.

**Files in this repo**
- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies
- `sample_reviews.csv` — sample data

Note: Large model files (`*.pth`) are excluded from the repo. See "Models" below.

**Quick setup (Windows PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Models**

This repository does not include the trained `.pth` model files. You have three recommended options to host/distribute them:

1. Host on Hugging Face Hub (recommended for ML models).
2. Use Git LFS to store large files in this repo.
3. Upload to cloud storage (Google Drive / Dropbox / S3) and provide direct download links.

Below are instructions and helper scripts to download models locally.

**PowerShell helper**

Edit `download_models.ps1` and replace the `PUT_MODEL_URL_HERE` placeholders with real URLs, then run:

```powershell
.\download_models.ps1
```

**Python helper**

You can also set environment variables `MODEL70_URL` and `MODEL80_URL`, then run:

```powershell
pip install requests
python download_models.py
```

**If you prefer Git LFS**

```powershell
# install git-lfs (if not installed)
# choco install git-lfs -y
git lfs install
git lfs track "*.pth"
# then add/commit your .pth files and push
```

**Hugging Face (recommended)**

- Create an account at https://huggingface.co
- Install `huggingface_hub` and login: `huggingface-cli login`
- Create a model repo and upload your `.pth` files there. Then update this README with the model repo URL.

---

If you want, I can:
- Upload the models using Git LFS (I can guide you through installing it), or
- Prepare a Hugging Face repo and push the model files (you'll need to login), or
- Add the model URLs into `download_models.ps1`/`download_models.py` and push them for you.

Reply which you want next and I'll proceed.