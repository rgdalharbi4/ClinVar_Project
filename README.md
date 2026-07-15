# ClinVar Analysis Project

Predicting whether a genetic variant's clinical interpretation is **Conflicting** or **Concordant**, using the [ClinVar Conflicting Classifications dataset](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting) (Kaggle).

## Project Overview

ClinVar is a public database where clinical laboratories submit their interpretation of genetic variants (e.g. "benign", "pathogenic"). Different labs don't always agree on the same variant — this project explores what distinguishes variants with **conflicting** interpretations from ones where labs **agree**, and builds a model to predict that conflict.

**Questions explored:**
- Which variant characteristics (population frequency, pathogenicity scores, gene, consequence type) are associated with conflicting interpretations?
- Can a model predict, from a variant's features alone, whether it's likely to receive conflicting interpretations?

**Expected insight:** rare variants (low population frequency) and variants with higher computational pathogenicity scores (CADD) are more likely to be classified inconsistently across labs, since they are harder to evaluate with high confidence.

## Dataset

- **Title:** ClinVar Conflicting Classifications
- **Source:** [Kaggle](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting)
- **Description:** ~65,000 genetic variants from the NCBI ClinVar database, with 46 original features covering population allele frequencies (ESP/ExAC/1000 Genomes), computational pathogenicity scores (CADD, LoFtool), and gene/consequence annotations. The target (`CLASS`) marks whether clinical laboratories' interpretations of a variant conflict (1) or agree (0).
- **Why this dataset:** chosen out of interest in the bioinformatics field — it combines a real clinical problem with a genuinely imbalanced, messy, high-dimensional dataset.

## Data Cleaning Steps

Full reasoning for each step is documented in `notebooks/02_data_cleaning.ipynb`. Summary:

1. **Missing values:** dropped columns with >20% missing data; imputed the rest (median for numeric, mode for categorical, 0 for allele-frequency columns).
2. **Feature engineering:** decomposed the multi-label `MC` column into 11 binary flags; converted `CLNDISDB`/`CLNDN` into a disease-count numeric feature.
3. **Dropped redundant/leaking columns:** `CLNHGVS` (unique ID, leakage risk), `EXON`, position columns (`cDNA`/`CDS`/`Protein_position`), `CADD_RAW` (near-duplicate of `CADD_PHRED`), `origin`/`biotype` (near-zero variance); consolidated `AF_ESP`/`AF_EXAC`/`AF_TGP` into a single `mean_af`.
4. **Renamed columns** to a consistent lowercase format.
5. **Removed duplicates** (none found).
6. **Outlier handling:** IQR-capped `cadd_phred` and `clinical_disease_count`; left `mean_af`'s extreme values intact since they carry real predictive signal (log-transformed instead, for modeling only).
7. **Encoding:** one-hot for low-cardinality columns, ordinal for `impact` (has a real severity order), label encoding for high-cardinality columns.

## Key Results

- **Cleaned dataset:** 65,188 variants, reduced from 46 raw features to 30 cleaned features (missing values resolved, redundant/correlated columns consolidated, outliers capped).
- **Class balance:** ~75% Concordant / ~25% Conflicting.
- **Best model:** Random Forest (`class_weight='balanced'`) — ROC-AUC 0.79, 75% recall on the Conflicting class.
- **Strongest predictor:** mean allele frequency (`mean_af`), followed by CADD pathogenicity score (`cadd_phred`) — rarer, more pathogenic-looking variants are more likely to have conflicting interpretations.

## Limitations & Future Work

- **Label reflects lab behavior, not ground truth:** "conflicting" depends on which labs happened to submit interpretations for a variant, not an objective measure of pathogenicity — a variant can be "concordant" simply because only one lab ever classified it.
- **Heavy missingness in dropped columns:** several raw features were missing 60-99% of values and were dropped rather than imputed; some of that discarded data (e.g. `SIFT`/`PolyPhen` predictions) could carry real signal a future version could try to recover instead of dropping.
- **Label/ordinal encoding on high-cardinality columns:** `symbol`, `feature`, `codons`, and `amino_acids` were label-encoded, which imposes an arbitrary numeric order with no biological meaning. Tree-based models tolerate this reasonably well, but it isn't ideal.
- **Moderate model performance:** ROC-AUC 0.79 is a solid baseline but not clinical-grade; no hyperparameter tuning or cross-validation was performed (a single train/test split), so the reported metrics carry some uncertainty.

**Recommendations for future analysis:** hyperparameter tuning with cross-validation, testing gradient-boosted models (XGBoost/LightGBM) against the Random Forest baseline, and revisiting the dropped high-missingness columns with more targeted imputation instead of removal.

## Dashboard Features

The Streamlit dashboard (`app.py`) is a dark-themed, interactive explorer built on the cleaned dataset and the trained model.

**Sidebar filters** (batched into a form — adjust several, then hit "Apply Filters" for one refresh):
- Chromosome (multiselect)
- Impact: MODIFIER/LOW/MODERATE/HIGH (multiselect)
- CADD_PHRED range (slider)
- Class: Concordant/Conflicting (multiselect)

**Sections:**
- **Overview** — filtered data preview, summary statistics, and a class distribution donut chart
- **Explore** — univariate (histogram, boxplot, countplot) and bivariate/multivariate (scatter, bar, correlation heatmap) charts, switchable via dropdowns
- **Predict** — a live what-if tool: move sliders for CADD score, allele frequency, LoFtool, disease count, and impact, then get a real-time prediction from the trained Random Forest, shown on a confidence gauge
- **Insights** — key findings from the analysis as styled cards

## Project Structure

```
data/
  raw/                          <- data BEFORE any cleaning (original Kaggle CSV)
    clinvar_conflicting.csv

  processed/                    <- data AFTER cleaning
    cleaned_semi_processedv1.csv   (checkpoint: after missing-value handling)
    cleaned_semi_processedv2.csv   (checkpoint: after MC/CLNDISDB/CLNDN encoding + CLNHGVS leakage-column drop)
    cleaned_semi_processedv3.csv   (checkpoint: after redundant/correlated feature drops + origin/biotype drop)
    cleaned_semi_processedv4.csv   (checkpoint: after column renaming to lowercase)
    cleaned_semi_processed.csv     <- FINAL cleaned dataset (outliers capped, human-readable, used by EDA + dashboard)
    clinvar_model_ready.csv        <- FINAL fully-encoded dataset (one-hot/ordinal/label encoded, mean_af log-transformed, used for modeling)
    rf_model.pkl                   <- trained Random Forest model (trained on clinvar_model_ready.csv's feature set)
    model_features.pkl             <- feature column order expected by the model

notebooks/
  01_data_exploration.ipynb     <- loading raw data, shape/info/describe, missing values
  02_data_cleaning.ipynb        <- missing values, feature engineering, outlier handling, encoding (raw -> processed)
  03_data_visualization.ipynb   <- univariate/bivariate/multivariate EDA plots, incl. a pre-cleaning correlation
                                    heatmap that justifies dropped/consolidated columns
  04_modeling.ipynb             <- train/test split, class imbalance handling (with a class-weight chart),
                                    model training & evaluation

src/
  visuals.py                    <- shared plotting functions used by the Streamlit dashboard

app.py                          <- Streamlit dashboard
requirements.txt                <- Python dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

## Running the Notebooks

Open any notebook in `notebooks/` with Jupyter/JupyterLab and run top to bottom:

```bash
jupyter lab
```

Run them in order (`01` → `04`) since each stage builds on the CSV produced by the previous one.

## Author

**Raghad Alharbi** — Tuwaiq Academy, Data Science & AI Bootcamp
