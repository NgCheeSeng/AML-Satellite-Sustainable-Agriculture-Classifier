# Satellite Sustainable Agriculture Classifier

This repository contains the source code for an AML final project that builds a satellite image time-series and geospatial feature pipeline for predicting sustainable Malaysian agriculture land into three labels:

```text
low / moderate / high
```

The source code is hosted on GitHub:

```text
git@github.com:NgCheeSeng/AML-Satellite-Sustainable-Agriculture-Classifier.git
```

The dataset is hosted separately on Hugging Face:

```text
https://huggingface.co/datasets/Aki298/AML-Satellite-Imagery-Malaysia-Copernicus
```

## Project Structure

```text
raw_to_be_processed/
  <latitude>_<longitude>_<label>.mp4
  <latitude>_<longitude>.txt
  <latitude>_<longitude>_<label>.txt  # accepted fallback

data/
  raw/<label>/<latitude>_<longitude>_<label>/
    original_video.mp4
    timeline.txt
    gee_observations.csv
    gee_feature_metadata.json

  processed/<label>/<latitude>_<longitude>_<label>/
    frame_000__YYYY-MM-DD.png
    frame_metadata.csv
    gee_features.csv        # model X only
    gee_targets.csv         # future/t+1 targets only

  processed/sample_index.csv
  processed/image_timeseries_results.csv  # future image model output

notebooks/
  01_video_to_cropped_frames.ipynb
  02a_fetch_gee_observations.ipynb
  02b_engineer_features_targets.ipynb
  03_eda_and_feature_selection.ipynb
  04_image_timeseries_urban_growth_predictor.ipynb
  05_model_training.ipynb

src/
  preprocessing/process_raw_videos.py
  features/gee_features.py
  features/model_inputs.py
```

The `data/` and `raw_to_be_processed/` folders are intentionally ignored by Git. Dataset files should be pulled from Hugging Face, not committed to GitHub.

## Environment Setup

Use the project conda environment:

```powershell
conda activate aml
python -m pip install -r requirements.txt
```

Or run commands without activating:

```powershell
conda run -n aml python -m pip install -r requirements.txt
```


## Google Earth Engine Credentials

Create a local credentials file for the Earth Engine project id. This file is ignored by Git.

```powershell
Copy-Item config\gee_credentials.example.json config\gee_credentials.json
```

Edit `config\gee_credentials.json` and set:

```json
{
  "project_id": "your-gee-project-id"
}
```

`02a_fetch_gee_observations.ipynb` also accepts `GEE_PROJECT_ID` from the environment, but the local credentials file avoids hardcoding the project id in notebooks.

## Pull Dataset from Hugging Face

Download the dataset into the local `data/` folder:

```powershell
conda run -n aml python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Aki298/AML-Satellite-Imagery-Malaysia-Copernicus', repo_type='dataset', local_dir='data', allow_patterns=['raw/**','processed/**'])"
```

If `data/` already exists, Hugging Face updates matching downloaded files, but local files that no longer exist remotely may remain. For a clean refresh, move or remove the old local `data/` folder before downloading again.

## Pipeline Order

1. `01_video_to_cropped_frames.ipynb`
   - Archives MP4/timeline files under `data/raw/<label>/<sample_id>/`.
   - Writes cropped image frames under `data/processed/<label>/<sample_id>/`.
   - Rebuilds `data/processed/sample_index.csv`.

2. `02a_fetch_gee_observations.ipynb`
   - Slow Google Earth Engine stage.
   - Writes only raw `gee_observations.csv` and `gee_feature_metadata.json` under `data/raw`.
   - Does not engineer features or targets.

3. `02b_engineer_features_targets.ipynb`
   - Fast local Pandas stage.
   - Reads raw `gee_observations.csv` from `data/raw`.
   - Applies leakage-controlled imputation.
   - Writes per-sample `gee_features.csv` and `gee_targets.csv` under `data/processed`.
   - Does not create `gee_features_all.csv` or `gee_targets_all.csv`.

4. `03_eda_and_feature_selection.ipynb`
   - Reads per-sample features/targets for health checks, missingness, class balance, and correlations.
   - Does not train models.

5. `04_image_timeseries_urban_growth_predictor.ipynb`
   - Future image time-series model stage.
   - Current implementation only reads image sequences and validates the expected output schema.
   - Later output should be `data/processed/image_timeseries_results.csv`.

6. `05_model_training.ipynb`
   - Future final classifier stage.
   - Current implementation only reads and merges GEE features, GEE targets, and optional image results.
   - No model training is implemented yet.
~
## Important Rules

- Raw GEE observations stay in `data/raw` and are not modified by feature engineering.
- Feature imputation happens only in `02b_engineer_features_targets.ipynb`.
- `gee_features.csv` must never contain `target_`, `future_`, or `delta_1` columns.
- Future/t+1 values stay physically separated in `gee_targets.csv`.
- Features and targets are per-sample files only; combined all-sample CSVs are intentionally not generated.