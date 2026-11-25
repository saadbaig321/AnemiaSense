# AnemiaSense

AnemiaSense is a Streamlit research app with two image analysis modules.

| Module | Input | Output |
| --- | --- | --- |
| Anaemia | Colour image with the RBC background removed, in the aneRBC format | Existing classifier, Grad-CAM, approximate CBC estimates and model dashboard |
| Thalassemia PBS | Full peripheral blood smear microscope field at 1000× | Segmented candidate cells, nine predicted morphology classes, low-score flags, annotated image and CSV |

## Updated interface

The interface takes visual inspiration from [Locus — Genetic Testing Website Design by Yscale Studio](https://dribbble.com/shots/27854580-Locus-Genetic-Testing-Website-Design), adapted to AnemiaSense. It uses warm ivory backgrounds, bold black headings, crimson emphasis and buttons, original glossy blood-cell artwork, a slim navigation bar, and responsive metric grids. The home screen includes a compact overview strip, an editorial research section, red and white module cards, and a dark three-step workflow section.

- **Home and sign-in:** shared branding, redesigned module selection, and the existing authentication flow.
- **Anaemia:** six animated held-out metrics, charts in a matching crimson palette, the existing threshold playground, prediction/Grad-CAM workflow, and animated CBC estimates with measured errors.
- **Thalassemia PBS:** saved evaluation metrics on entry, source image preview, and animated field statistics after analysis. Existing candidate review and CSV export remain available.
- **Motion:** metric panels reveal and count up once they enter the viewport. Charts and supporting panels fade upward on scroll. Values use the saved evaluation or actual analysis result; animation does not change calculations.
- **Accessibility:** reduced-motion preferences disable movement and show final values immediately. Screen readers receive the final metric values without announcing every animation frame. Cards adapt from three to two to one column.

### Frontend files

| File | Purpose |
| --- | --- |
| `ui/__init__.py` | Shared hero, branding, section headings, and Streamlit v2 components |
| `ui/styles.css` | App-wide styling and responsive layouts |
| `ui/metrics.css` | Isolated metric-card styling |
| `ui/metrics.js` | Viewport observation, number animation, and cleanup |
| `ui/motion.js` | Scroll reveals for native Streamlit output |
| `ui/assets/blood-cells-hero.webp` | Original decorative blood-cell artwork |
| `.streamlit/config.toml` | Matching theme for native widgets |

No Node build step is required. CSS, JavaScript, artwork, and branding are bundled locally. The decorative artwork is packaged as a compact WebP; it is not used as model input. Keep the pinned Streamlit version: the animation components use its v2 component API. All model checkpoints and inference code are retained.

## Upgrade the existing Streamlit app

This project can replace the anaemia-only `main` branch of `saadbaig321/AnemiaSense`. The repository also has a `fyp-ii-thalassemia-pbs` branch containing the earlier two-module project. The redesigned interface is in this update package.

1. Confirm which branch your existing Streamlit deployment uses. If it uses `main`, upload the update package's extracted files into that branch's repository root, alongside the existing `app.py`.
2. Preserve the folder structure, including `ui/`, `.streamlit/`, and the two new PBS files in `dashboard_data/`. Existing anaemia checkpoints are unchanged and can remain in the repository.
3. Commit the complete update together, including `requirements.txt`. The older repository pins Streamlit 1.39.0; the updated interface requires the bundled Streamlit 1.67.0 dependency.
4. Keep the existing deployment's repository, branch, and `app.py` entry point. Streamlit then rebuilds that app using the updated source.
5. The updated project includes sign-in. Configure the OpenID Connect settings in Streamlit secrets as described below if they are not already configured. The local demo environment variable is intended for local demonstrations.

## Run locally

Use Python 3.11 or 3.12.

```sh
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For a local FYP demonstration, explicitly enable demo mode.

```sh
# macOS/Linux
ANEMIASENSE_DEMO=1 python -m streamlit run app.py --server.address=127.0.0.1
```

```powershell
# Windows PowerShell
$env:ANEMIASENSE_DEMO = "1"
python -m streamlit run app.py --server.address=127.0.0.1
```

### Enable sign-in

1. Create a web client in your OpenID Connect provider. The example uses Google.
2. Register `http://localhost:8501/oauth2callback` for local use, or `https://YOUR-APP/oauth2callback` on the host.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`. Set the matching redirect URI, provider metadata URL, client ID and client secret. Generate a cookie secret with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. On a managed deployment, enter those values in its secrets settings.
4. Start with `python -m streamlit run app.py`. Once signed in, choose either module on the first screen. Use **Choose another module** to return to the chooser.

Configured sign-in takes precedence over demo mode. Keep the secrets file out of version control.

## PBS methodology and model

The module implements the FYP II report's preprocessing sequence with OpenCV and scikit-image. It fits the image to 800 × 600 while preserving aspect ratio, extracts the green channel, median-filters it, detects edges, dilates and fills the mask, separates touching regions with distance-transform watershed, erodes, then excludes small and border regions. Area, intensity and eccentricity filters select candidate cells. These filters are implementation choices and the boundaries are approximate.

A small CNN classifies the masked green-channel crops. The included checkpoint was trained on 5,686 of 7,108 official labelled crops. Its test set contains 1,422 crops from 355 held-out source fields, grouped by source field identifier. Macro F1 was 0.872 and balanced accuracy was 87.6% on the held-out crops. The test split can include fields from the same patient. The pencil-cell class has five test crops. The evaluation does not measure segmentation accuracy, independent-patient performance or patient-level diagnosis.

To retrain, download the nine ZIP archives from [RBCdataset v1](https://doi.org/10.17632/rfdz6wfzn4.1) into one directory and run:

```sh
python scripts/train_pbs.py path/to/rbc-archives
```

The bundled demonstration image is `T1-001.png` from [ThalassemiaPBS v1](https://doi.org/10.17632/gd9ysj73jd.1), resized and re-encoded as JPEG. Both datasets are by Dyah Aruming Tyas, Tri Ratnaningsih, Agus Harjoko and Sri Hartati, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the [dataset paper](https://doi.org/10.1016/j.dib.2022.107886).

The PBS workflow is for research and educational use. Clinical diagnosis requires clinical assessment and appropriate laboratory testing.
