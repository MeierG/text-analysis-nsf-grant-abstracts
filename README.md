# Text Analysis of NSF Awards Abstracts
==============================

Objective: predicting grant award amounts based on text patterns.

One of the biggest concerns for private individuals, researches, and non-profit organizations is to get funding to continue and expand their work. Although there are many options to obtain funding and organizations rarely depend on just one source, government grants provide one of the best alternatives to finance research and projects with a socio-economic impact.

Since grants funded by the U.S. government are one of the most common sources of funding and preparing a proposal can be a daunting process, knowing the project topics that get the most funding and the terms that are most likely related to greater financial support, can help individuals and organizations find the best ways to submit their plans based on theme, length, and amount of money needed.

Hence, the goal of this project is to use publicly available award data from the U.S. National Science Foundation (NSF) to identify whether the type of language used and linguistic patterns have have an impact on the aid received. The analysis considers the length of the abstract, ease of read and other linguistic patterns as predictors and the results can help future applicants understand how their linguistic patterns can impact the funding received based on theme, length,number of words and proportion of adverbs used.

## The final interactive Jupyter notebook can be seen on <a href = "https://nbviewer.jupyter.org/github/MeierG/text-analysis-nsf-grant-abstracts/blob/master/notebooks/nsf-grants.ipynb">nbviewer.</a>

## The final report <a href = "https://github.com/MeierG/text-analysis-nsf-grant-abstracts/blob/master/reports/nsf-grants-text-analysis-and-funding-predcition.pdf">on PDF format can be found here.</a>

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original data exceeds the size limit for github. File with direct link instead.
    │
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
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
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
