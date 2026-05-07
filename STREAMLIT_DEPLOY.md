# Deploying BVID-FE on Streamlit Community Cloud

This is the one-time setup to put the BVID-FE web app on the public
[share.streamlit.io](https://share.streamlit.io) hosted tier so fellow
engineers can use it via a URL.

The Streamlit deploy is **additive** — it does not replace the desktop
PyQt GUI (`bvidfe-gui`), the CLI (`bvidfe`), or the PyInstaller bundles
shipped via the GitHub Actions release workflow. All four entry points
share the same `BvidAnalysis` engine.

## Prerequisites

- A GitHub account that can read this repository.
- A Streamlit Community Cloud account (free; sign in with GitHub).

## One-time deployment

1. **Push `app.py` and `requirements.txt` to a branch on GitHub.**
   They were committed alongside this guide, so:

   ```bash
   git push
   ```

   The cloud worker reads from whichever branch you point the app at
   (typically `main`); confirm `app.py` and `requirements.txt` are at
   the repo root on that branch.

2. **Sign in to Streamlit Cloud.**
   Go to <https://share.streamlit.io>, click **Sign in**, and authorize
   the Streamlit GitHub app for this repository. You may need to ask a
   GitHub org admin to install the integration if the repo lives in an
   organization.

3. **Click "New app".**
   Fill in the form:

   | Field | Value |
   |---|---|
   | Repository | `<owner>/BVID-FE` |
   | Branch | `main` (or whichever branch holds `app.py`) |
   | Main file path | `app.py` |
   | Python version | `3.11` (matches the CI matrix; 3.12 also works) |
   | App URL | leave the default `<repo>.streamlit.app`, or set a custom subdomain |

   Streamlit will detect `requirements.txt` automatically from the repo
   root. The `.` line at the top of `requirements.txt` triggers
   `pip install .`, which installs the `bvidfe` package itself from
   `pyproject.toml`.

4. **Click "Deploy".**
   First build takes 3–5 minutes (numpy + scipy wheels are large). The
   build log is visible in the cloud UI; if dependencies fail to
   install you'll see the error there. Once the build succeeds the app
   is live at the URL you chose.

5. **(Optional) Restrict access.**
   On the free tier the app is public-by-URL. To restrict to specific
   GitHub users, open the app's **Settings → Sharing** and add them as
   viewers. This requires each viewer to also have a Streamlit Cloud
   account.

## Updating the deployed app

Push to the configured branch. Streamlit Cloud auto-redeploys on every
push to that branch — typical turnaround is 1–2 minutes for a
code-only change, longer if `requirements.txt` changes (because the
deps are reinstalled from scratch).

Manual redeploy: in the cloud UI click **Manage app → Reboot app**.

## Limits to know about (Streamlit Cloud free tier)

- **1 GB memory** per worker. The app's pre-flight warns when an
  `fe3d` configuration would exceed ~100 k DOFs; over that, the worker
  may be OOM-killed and the user will see an "app crashed" page. Set
  `BVIDFE_FE3D_MAX_DOF` to a smaller value (e.g. `200000`) via the
  Streamlit Cloud **Secrets** tab if you want a hard ceiling — this
  surfaces a Python-level `FE3DSizeError` instead of a hard-kill.
- **1 concurrent worker** per app on the free tier — multiple
  simultaneous users share the same Python process. Long fe3d runs
  block other users.
- **Apps go to sleep** after a period of inactivity. First request
  after sleep takes ~60 s to wake.
- **No persistent storage.** Each session gets a fresh Linux container.
  Result downloads (the **Export** tab) are the user's responsibility
  to save.
- **No GPU.** The fe3d sparse solver is CPU-only and that's fine for
  the existing problem sizes.
- **No PyVista 3D.** The cloud worker has no display server, so the
  app intentionally falls back to matplotlib-only 2D plots. Users who
  need the full 3D view should run the desktop GUI locally.

## Local sanity-check before deploying

```bash
pip install -r requirements.txt
streamlit run app.py
```

The local Streamlit server runs on `http://localhost:8501` and behaves
identically to the cloud version — confirm the inputs, the run, and
the result tabs all work before pushing.

## Troubleshooting

- **`ModuleNotFoundError: bvidfe`** → the `.` line in `requirements.txt`
  was removed or the repo's `pyproject.toml` failed to build. Check
  the cloud build log for the `pip install .` step.
- **`pip install .` builds slowly** → expected on first deploy and
  after every `requirements.txt` change. Subsequent deploys reuse the
  pip cache and rebuild only the local package.
- **The app crashed** banner with no traceback → almost always an OOM
  during fe3d. Reduce mesh resolution or pin a smaller
  `BVIDFE_FE3D_MAX_DOF` via secrets.
- **PyQt-related ImportError** → something imported `bvidfe.gui.*`
  from `app.py` directly. The cloud `requirements.txt` does not
  include PyQt6 by design. Keep the GUI imports in the desktop entry
  point only.
