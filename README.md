# MASSTER

## Project Description
Application of active-learning and semi-supervised learning to multi-target regression.

Included active-learning models:
* Instance-Based:
    * Instance-Based
    * Random Sampling
    * Baseline method: Greedy Sampling 
    * RT-AL: Regression-Tree based Active-Learning 
* Target-Based:
    * MASSTER-AL
      
Included semi-supervised methods:
* Self-Learning
* Co-training
* PCT

Included proposed methods ASSL:
* MASSTER 

### Sources:
https://github.com/QuintenDanneels/AL-for-MTR
https://github.com/AshnaJose/Regression-Tree-based-Active-Learning


## Project Pipeline
### Active Learning only
1. On the main branch, import .csv dataset into data/raw directory
2. Update src/config.py file

    2.1. Replace DATASET_NAME variable by the dataset file name e.g. DATASET_NAME = 'atp7d'
    2.2. Define other relevant variables, such as K_FOLDS, N_EPOCHS, BATCH_PERCENTAGE, N_TREES, etc
   
3. Run src/data/data_processing.py to generate train (labeled), pool (unlabeled) and test (labeled) datasets
4. Run src/models/active_learning/original/active_learning_module.py to apply active-learning models
5. Evaluate what active-learning model generated the best performance metrics for all datasets (use autorank to evaluates its statistical relevance)
6. Choose the best active-learning model

### Semi-Supervised only
1. On the main branch, import .csv dataset into data/raw directory
2. Update src/config.py file

    2.1. Replace DATASET_NAME variable by the dataset file name e.g. DATASET_NAME = 'atp7d'
    2.2. Define other relevant variables, such as K_FOLDS, N_EPOCHS, BATCH_PERCENTAGE, N_TREES, etc
   
3. Run src/data/data_processing.py to generate train (labeled), pool (unlabeled) and test (labeled) datasets
4. Run src/models/semi_supervised_learning/self_learning.py to apply self-learning
5. Run src/models/semi_supervised_learning/cotraining.py to apply co-training
6. Evaluate SSL models performance

### MASSTER
1. On the main branch, import .csv dataset into data/raw directory
2. Update src/config.py file

    2.1. Replace DATASET_NAME variable by the dataset file name e.g. DATASET_NAME = 'atp7d'
    2.2. Define other relevant variables, such as K_FOLDS, N_EPOCHS, BATCH_PERCENTAGE, N_TREES, etc
   
3. Run src/data/data_processing.py to generate train (labeled), pool (unlabeled) and test (labeled) datasets
4. Run src/models/proposed_method/masster.py to run MASSTER-CT and MASSTER-AL
5. Evaluate ASSL performance using src/evaluation/evaluation_paper.py

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
│  
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         template and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures         <- Generated graphics and figures to be used in reporting
|   └── active_learning_only <- Generated results for active-learning module for diferent active-learning models
|   └── active_learning <- Generated results for active-learning module for MASSTER-AL
│   └── paper_evaluation <- Generated evaluation results from the proposed ensemble models
    └── semi_supervised_learning <- Generated results for semi_supervised_learning models
|
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    └──  data
         └── data_processing.py        <- Process raw data into train, test and pool
         └── dataframes_creation.py        <- Transform data into dataframes
    └──  evaluation
         └── evaluation_active.py        <-Generate reports for active learning only
         └── evaluation_paper.py        <- Generate reports for paper analysis comparing proposed models
│
├── models                <- Store useful variables and configuration
   └── active_learning        <-Active Learning models
            └── original        <- All active_learning models implemented in numpy
            └── active_learning.py <- active_learning in pandas
            └── target_qbc.py <- MASSTER-AL inplemented in pandas
   └── proposed_method        <- MASSTER model
         └── masster.py       <- MASSTER model
   └── semi_supervised_learning        
         └── cotraning.py       <- co-training model
         └── self_learning.py       <- self-learning model
│
├── utils               <- Useful functions to help the readability of the code
      └── aux_active  <-Functions used during active_learning
      └── data_handling  <-Read and save csv files
      └── metrics  <- Functions to calculate metrics

├── config.py               <- Store useful variables and configuration
--------

