# Multivariate Time Series Transformer Framework
Multivariate Time Series Transformer, Python 3.12+ and Apple Silicon Implementation

This version solved the old Python 3.8 dependency and added support for Apple Silicon.


## Setup
`cd mvts_transformer/`

`pip install -r failsafe_requirements.txt`



### Test without modifying code

The major code that needs some customizations are `datasets/data.py`, `src/main.py`.

It is also suggested to create your own scripts side-by-side to this repo:

ParentFolder/
│
├── mvts_transformer_M
│
└── run/
    ├── data.py             # with customizations
    ├── main.py             # with customizations
    ├── variables.py        # organize the variables
    └── cust_run.py         # a script to call the main.py



### Adding your own datasets

To train and evaluate on your own data, you have to add a new data class in `datasets/data.py`.
You can see other examples for data classes in that file, or the template in `example_data_class.py`.

The data class sets up one or more `pandas` `DataFrame`(s) containing all data, indexed by example IDs.
Depending on the task, these dataframes are accessed by the Pytorch `Dataset` subclasses in `dataset.py`.

For example, autoregressive tasks (e.g. imputation, transduction) require a member dataframe `self.feature_df`, 
while regression and classification (implemented through `ClassiregressionDataset`) additionally require a `self.labels_df` member
variable to be defined inside the data class in `data.py`.

Once you write your data class, you must add a string identifier for it in the `data_factory` dictionary inside `data.py`:
```python
data_factory = {'weld': WeldData,
                'tsra': TSRegressionArchive,
                'pmu': PMUData,
                'mydataset': MyNewDataClass}
```


## Origin Paper
This code corresponds to the [paper](https://dl.acm.org/doi/10.1145/3447548.3467401): George Zerveas et al. **A Transformer-based Framework for Multivariate Time Series Representation Learning**, in _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14-18, 2021_.
ArXiV version: https://arxiv.org/abs/2010.02803