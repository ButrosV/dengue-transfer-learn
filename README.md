# Dengue Transfer Learning


***

## Task

Build TensorFlow transfer learning pipeline to forecast **weekly dengue cases** (`total_cases`) from 22 multivariate weather/environmental features, including full support for:

- City-specific data preprocessing and sliding window generation
- Transfer learning: pretrain San Juan → finetune Iquitos
- Hyperparameter tuning with Keras Tuner (50 trials) on 4GB GTX 960M GPU
- Model evaluation comparing scratch vs transfer learning baselines

Goal: Demonstrate **25% MAE improvement** via transfer learning on small epidemiological dataset (2 cities, ~1500 labeled rows).

## Project Structure

```
.
├── .gitignore                   	# Excludes data/models from GitHub
├── docker-compose.yml           	# GPU training container (TF 2.12)
├── environment.yml              	# mamba preprocessing env (Python 3.11)
├── README.md                    	# This file
│
├── data/                        	# Local only (.gitignore protected)
│   ├── raw/                     	# dengue_features_train.csv, dengue_labels_train.csv
│   ├── processed/               	# sj_train.parquet, iq_val.parquet
│   └── intermediate/            	# sj_windows.npy, iq_windows.npy
│
├── notebooks/				# EDA + results (committed to GitHub)
│   ├── 01_eda_dengue.ipynb		# City splits, feature correlations,
│   ├── 02_data_cleaning_dengue.ipynb
│   ├── 03_feature_eng_dengue.ipynb
│   └── 04_feature_selection.ipynb	# final feature selection, Walk-forward Cross-Validation, pre-modeling preprocessing
│
├── src/				# Modular Python code (committed to GitHub)
│   ├── config.py
│   ├── models.py			# model_builder() for Keras Tuner
│   ├── preprocessing/
│   │   ├── clean.py
│   │   ├── engineer/
│   │   │   ├── base.py
│   │   │   ├── pipeline.py
│   │   │   └── temporal.py
│   │   ├── pipeline.py
│   │   ├── preprocess.py		# create_windows
│   │   └── select.py
│   └── utils/
│       ├── eda.py
│       ├── utils.py			# load and save data, mae_table() ??, misc
│       └── visualizations.py		# plot_results()
│
├── tensorman/                   	# Docker training scripts
│   ├── Dockerfile               	# TF 2.12.0-gpu + keras-tuner 1.4.7
│   └── train.py                 	# tuner.search() hyperparameter optimization
│
├── models/                      	# Trained models (.gitignore protected)
│   └── best_transfer.h5
└── reporting/                   	# Portfolio screenshots (committed to GitHub)
    └── pics/
        ├── weather_features_grid.png
        └── mae_comparison.png
```

## Dual Environment Architecture

**Dual-stack workflow** separating CPU preprocessing from GPU training:

**`tensorman/`**: Docker **GPU training** environment (GTX 960M optimized)

- Originally used Tensorman TF images
- Now uses `tensorflow/tensorflow:2.12.0-gpu` (more reliable)
- Contains `Dockerfile` + `train.py` for Keras Tuner hyperparameter tuning

**Mamba python-environment** (or any other python environment): CPU-only **EDA + preprocessing + evaluation** environment

- Python 3.11 + pandas 2.2.* + JupyterLab 4.*
- EDA (`01_eda.ipynb`), window generation (`02_windowing.ipynb`)
- Creates NumPy arrays for Docker training (`data/intermediate/*.npy`)

```
Workflow: mamba (EDA → .npy) → Docker (train → .h5) → mamba (results → plots)
```

## Usage

#### 1. Install Dependencies

Clone the [git repository](https://github.com/ButrosV/dengue-transfer-learn) or download project structure.

##### 1.1. Preprocessing Environment Setup (CPU-only mamba)

Create analysis environment for EDA and window generation:

```bash
# Create mamba environment
mamba env create -f environment.yml -n your-env-name
```

```bash
# alternatively use pip in your python==3.11 environment
pip install -r requirements.txt
```

Copy dengue datasets to `data/raw/` and run preprocessing:

```bash
cd ~/dengue-transfer-learning
jupyter lab
# Run notebooks/01_eda.ipynb → Explore 22 features × 2 cities
# Run notebooks/02_windowing.ipynb → Generate sj_windows.npy, iq_windows.npy
```


##### 1.2. Training Environment Setup (GPU Docker)

Build and run TensorFlow 2.12 GPU container:

```bash
# First time: Build TF 2.12 + keras-tuner (!!! 10GB image !!!)
docker compose up --build -d

# Verify GPU + data mount
docker compose exec dengue-tf bash
```

**Train with hyperparameter tuning:**

```bash
docker compose exec dengue-tf bash
cd /workspace
python tensorman/train.py  # Keras Tuner 50 trials → models/best_transfer.h5

# OR one-command training
docker compose run --rm dengue-tf python tensorman/train.py
```

**Access Jupyter inside container:**

```
http://localhost:8888  # TF 2.12 JupyterLab with GPU
```


#### 2. Results Analysis

Exit container and analyze locally:

```bash
docker compose down
mamba activate your-env-name  # or use your non-mamba environment
jupyter lab
```


### Clean up your working directory

- **Stop container and clean Docker:**

```bash
docker compose down
docker system prune -f  # Remove unused images/containers
```

- **Remove local data/models (keep code):**

```bash
rm -rf data/* models/*
```

- **Full reset (keep git-tracked files only):**

```bash
git clean -fdx  # Removes ALL untracked files/folders
```


## Requirements

**Preprocessing (mamba `your-env-name` or your non-mamba environment):**

```txt
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
jupyterlab>=4.0.0
ipywidgets>=8.0.0
polars>=1.0.0
matplotlib
seaborn
plotly
fastparquet
python-kaleido
python=3.11
```

**Training (Docker `dengue-tf212:v1`):**

```txt
tensorflow==2.12.0-gpu
keras-tuner==1.4.7
Docker + NVIDIA Container Toolkit
```

