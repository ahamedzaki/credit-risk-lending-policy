# Deploy the Lending Policy Simulator (Streamlit Community Cloud)

The app is self-contained: it reads only `exports/simulator_base_sample.parquet`
(a committed ~1 MB, 50k-row sample) and `config.yaml`. No database, no model file.

## Prerequisites
- Repo pushed to GitHub (see below).
- `requirements.txt` at repo root = the slim app deps (already set).
- `exports/simulator_base_sample.parquet` committed (it is; `.gitignore` has an exception).
- Entry point: `app/simulator.py`.

## One-time: push to GitHub

```bash
cd ~/Desktop/credit-risk-lending-policy
gh repo create credit-risk-lending-policy --source=. --push --private   # or --public
```

If the repo already exists:

```bash
git remote add origin https://github.com/<you>/credit-risk-lending-policy.git
git push -u origin main
```

## Deploy

1. Go to <https://share.streamlit.io> → sign in with GitHub.
2. **New app** → pick the repo, branch `main`, **Main file path** `app/simulator.py`.
3. **Advanced settings → Python version: 3.11**.
4. **Deploy.** First build installs `requirements.txt` (~1–2 min).
5. You get a URL like `https://<name>.streamlit.app` — put it in the README and your CV.

Private repo? Streamlit will ask to install its GitHub app and grant read access to that
repo — that is enough, no need to make it public.

## Refreshing the deployed data

The committed sample is a 50k-row reservoir sample (seed 42) of the full 640k-loan
`mart_simulator_base`. To refresh it after a pipeline change:

```bash
python run_pipeline.py export          # rewrites exports/simulator_base_sample.parquet
git add exports/simulator_base_sample.parquet && git commit -m "refresh app sample" && git push
```

Streamlit Cloud redeploys automatically on push.

## Sanity check after deploy
The deployed app loads the 50k **sample**, so dollar totals are ~13× smaller than
`reports/memo.md` (which is the full 640k book). The **rates** must match: at the default
policy (PD < 0.15, FICO ≥ 660) approval ≈ 64%, expected default rate ≈ 9%, and the
profit-vs-approval curve still peaks near PD < 0.17. The caption shows which file loaded
(`simulator_base_sample.parquet` when deployed).
