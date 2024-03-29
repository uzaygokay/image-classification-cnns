Image classification with CNNS
==============================

This project aims to classify images from CIFAR-10 dataset by using various CNN architectures with Pytorch Lightning.

### To use the code ###
1) Clone the repo
2) Setup a virtual env by using environment.yaml
3) Change directory to src/models/
4) Run the train_model.py file

* Modify model name according to your choice

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── arguments.py       <- Hyperparameters and arguments for model.
    │   │   ├── datamodule.py      <- Reads, transform and loads the data.
    │   │   ├── logger.py          <- Logger to track the run.
    │   │   ├── model.py           <- Script for image classifier.
    │   │   ├── alexnet.py         <- Implementation of AlexNet.
    │   │   └── train_model.py     <- Script to train models.
    │   │
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
