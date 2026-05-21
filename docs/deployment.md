# Deployment

BVID-FE ships with a Streamlit web app (`app.py`) that wraps
`bvidfe.analysis.BvidAnalysis`. This page summarises how to publish a public
URL on Streamlit Community Cloud — the full guide lives in
[`DEPLOYMENT_STREAMLIT.md`](https://github.com/ranipdx-glitch/BVID-FE/blob/main/DEPLOYMENT_STREAMLIT.md).

## Local smoke test

```bash
pip install -r requirements.txt
pip install -e .
streamlit run app.py
```

A browser tab should open at <http://localhost:8501>. Tweak parameters in
the sidebar, click **Run analysis**, and verify the **Results** tabs
populate.

## Streamlit Community Cloud

1. Push your branch (typically `main`) to GitHub.
2. Sign in at <https://share.streamlit.io> with the GitHub account that owns
   the repo (`ranipdx-glitch`).
3. **Create app** → **Deploy a public app from GitHub**:
    - **Repository:** `ranipdx-glitch/BVID-FE`
    - **Branch:** `main`
    - **Main file path:** `app.py`
    - **App URL:** pick a sub-domain (e.g. `bvidfe.streamlit.app`)
4. In **Advanced settings** set Python to **3.11**.
5. Click **Deploy**. First build takes 2–4 minutes; subsequent redeploys
   finish in ~30 s.

Every push to the configured branch triggers an automatic redeploy.

## Resource limits

Streamlit Cloud (Community tier) provisions per app:

- 1 GB RAM
- 1 vCPU
- shared, ephemeral filesystem
- no GPU
- public repos only by default

Implications for BVID-FE:

- The **fe3d** tier solves a 3D FE problem and can easily blow past 1 GB
  on dense meshes. The Streamlit app defaults to **empirical** and warns
  before launching fe3d. If you allow fe3d in the cloud, keep
  `elements_per_ply = 1` and `in_plane_size_mm >= 5`.
- Don't write user-supplied data anywhere except `tempfile`-scoped paths;
  the filesystem resets on every reboot.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Build fails on `PyQt6` / `pyvista` | They should not be in `requirements.txt`. Streamlit Cloud has no display server. |
| App boots but `import bvidfe` fails | Confirm the package installs from source: `pip install -e .` works locally. |
| Plots don't render | Make sure `matplotlib.use("Agg")` is called before any `pyplot` import. `app.py` already does this. |
| Slow first run | Expected — numpy/scipy/shapely wheels are ~100 MB combined. |
| Out of memory on large fe3d run | Increase `in_plane_size_mm`, keep `elements_per_ply = 1`, or switch to the empirical / semi_analytical tier. |
| `FE3DSizeError` | The mesh exceeds `BVIDFE_FE3D_MAX_DOF`. Coarsen the mesh — Streamlit Cloud doesn't have the headroom to raise this cap. |

## Optional next steps

- Add a `runtime.txt` with `python-3.11` to pin the Python version exactly.
- Configure a custom domain in Streamlit Cloud's settings.
- Add Streamlit secrets via the Manage panel if the app later needs API keys.
- Add CI to run `streamlit run --headless` smoke tests on every PR.
