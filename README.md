# Chicago Crimes ML Project (2001-2018)

## Overview
This project analyzes crime data from Chicago spanning 2001 to 2018. The goal is to predict whether a crime incident results in an arrest using machine learning models. The dataset includes details like crime type, location, time, and administrative information.

Key findings:
- Most crimes do not lead to arrests.
- Arrest rates vary by crime type, location, and time.
- Random Forest outperforms Logistic Regression in prediction accuracy.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle: https://www.kaggle.com/datasets/spirospolitis/chicago-crimes-20012018-november
4. Ensure the data file `city_of_chicago_crimes_2001_to_present.csv` is in the `data/` folder.

## Usage
Run the notebooks in order:
1. `01_EDA.ipynb` - Explore the dataset.
2. `02_data_preparation.ipynb` - Clean and prepare data.
3. `03_modeling.ipynb` - Train models.
4. `04_evaluation.ipynb` - Evaluate models.

Trained models are saved in `models/` and can be loaded for predictions.

## Project Structure
- `data/` - Raw and processed datasets.
- `models/` - Saved trained models (Logistic Regression and Random Forest).
- `notebooks/` - Jupyter notebooks for analysis.
- `utils/` - Helper functions for preprocessing and visualization.

## Notebooks
- **01_EDA.ipynb**: Exploratory Data Analysis - Understand data, identify patterns, and select features.
- **02_data_preparation.ipynb**: Data cleaning, feature engineering, encoding, and splitting.
- **03_modeling.ipynb**: Train baseline (Logistic Regression) and improved (Random Forest) models.
- **04_evaluation.ipynb**: Compare models using metrics, confusion matrices, and ROC curves.

## Authors
- Cherifa Ben Ghorbel
- Chahd Fadhloui



