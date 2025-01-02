# Active and Self-Learning for Multi-target Regression

Template structure for data science projects from cookiecutter

## Project Description
Application of active-learning and semi-supervised learning to multi-target regression.

Included active-learning models:
* Instance-Based:
    * Instance-Based
    * Random Sampling
    * Baseline method: Greedy Sampling 
    * RT-AL: Regression-Tree based Active-Learning 
* Target-Based:
    * QBC-RF
      
Included semi-supervised methods:
* Self-Learning
* Co-training
* PCT

### Sources:
https://github.com/QuintenDanneels/AL-for-MTR
https://github.com/AshnaJose/Regression-Tree-based-Active-Learning

## Project Pipeline
1. On the main branch, import .csv dataset into data/raw directory
2. Update src/config.py file

    2.1. Replace DATASET_NAME variable by the dataset file name e.g. DATASET_NAME = 'atp7d'
    2.2. Define other relevant variables, such as K_FOLDS, N_EPOCHS, BATCH_PERCENTAGE, N_TREES, etc
   
3. Run src/data/data_processing.py to generate train (labeled), pool (unlabeled) and test (labeled) datasets
4. Run src/models/active_learning_module.py to apply active-learning models
5. Evaluate what active-learning model generated the best performance metrics for all datasets (use autorank to evaluates its statistical relevance
6. Choose the best active-learning model
7. Run a combination of the chosen active-learning with each semi-supervised metric
8. Evaluate the results

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         template and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting
|   └── active_learning <- Generated results for active-learning module
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── template   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes template a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

