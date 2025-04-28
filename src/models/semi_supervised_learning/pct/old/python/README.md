## What is PyClus

PyClus is a Python wrapper around [CLUS](https://github.com/knowledge-technologies/clus).
For all the information about CLUS itself (what are the parameters, settings files, etc.) please, 
follow the link above and consult the manual.

The rest of this readme describes PyCLUS and assumes you are (somewhat) familiar with CLUS.

## Is PyClus for me? 

PyClus is for you if you like scikit, but - at the same time - you are tired of

- manipulating nominal attributes in scikit,
- manipulating missing values in scikit,
- manually computing label weights for hierarchical multi-label classification.

PyClus is for you if you would like to use scikit, but you have to use CLUS.

PyClus is NOT
- for you if you want to use Python but still want to get the output file and everything after one line,
as is achieved by `> java clus.jar file.s`.
- your solution for the issues you are facing when running CLUS
(weird errors, empty output files, etc.).

# Dependencies

We need 

- java, 
- `numpy` and `scikit learn`.

# How to install it?

Simply call 

`pip install pyclus`

# Usage

Arguably, PyClus is somewhat simpler to use than CLUS itself:

```
# define xs_train, y_train, xs_test, y_test ...
model = ClassificationTree(verbose=0, is_multi_target=False, min_leaf_size=12)
model.fit(xs_train, y_train)
y_hat_train_all = model.predict(xs_test)
```

We can see that PyClus decouples CLUS's rigid pipeline of 

- loading the data,
- learning the models,
- evaluating the models,
- outputting everything

executing only one command.


## Loading the data

The allowed data formats for *features* are:

- `np.ndarray` (2D)
- `List[List[Any]]`
- `Tuple[List[int], List[int], List[Any]]` (list of row indices, list of column indices and list of values)

The first two are for dense data, the third one is for sparse data.

Missing values should be denoted by the string `'?'`.

The allowed data formats for *targets* are:
- `np.ndarray` (1D and 2D),
- `List[List[Any]]` (multiple targets or single target as 1D multiple target),
- `List[Any]` (single target).

For targets, the sparse format is not allowed.

## Model initialization

PyClus defines a few classes of models that can be used for

- classification
- regression
- multi-label classification (MLC), and
- hierarchical multi-label classification (HMLC)

tasks.

For each of this problems, two classes are defined:

- `<task>Tree` (e.g., `RegressionTree`, `HMLCTree`)
- `<task>Ensemble` (e.g., `ClassificationEnsemble`, `MLCEnsemble`)

We initialize the objects in a scikit-fashion, however, there are some differences due to CLUS's peculiarities
and immense number of options/parameters.

### Settings file parameters

The most common arguments among those that are in CLUS passed to `.s` file
(e.g., `n_trees`, `min_leaf_size`, `ensemble_method`, etc.) are listed explicitly and
can be seen in the signature of the class constructor.

All of them have default values.

Those that are used less frequently, (e.g., `SplitPosition` in Section  `Tree` of the settings file),
can be passed to constructors as part of `**kwargs` as  `<Section>_<Parameter>=value`, for example

`model = ClassificationTree(..., Tree_SplitPosition='Middle')` 

If possible, do not use `Output` section (that might mess up fit and/or predict).

**IMPORTANT NOTES**:
- CLUS allows you to learn a single tree and still specify any number of trees in Section Ensemble 
 of the settings. PyClus is not that liberal.
- If you pass both `n_tree=100` and `Ensemble_Iterations=10`, the kwarg value wins, i.e., we will learn 10 trees.
The same goes for other explicitly named parameters.
- If your clustering attributes are not the same as your target attributes, note that the temporary arff lists
the attributes in the following order: `x1, x2, xN, y1, ..., yN`.

### CLUS command line switches

If you are not using simply `clus.jar file.s` but rather 

`clus.jar -forest -ssl file.s`,

add a kwarg for each switch to the constructor as `<switch>=value`, where
`value` is the list of arguments for the switch. If the switch takes no arguments
 (as, for example `ssl`), use `[]` (empty list).
 
A pyclus analogue of the java call above would be (for MLC data)

`model = MLCEnsemble(..., forest=[], ssl=[])`. 


**IMPORTANT NOTES**:

In contrast to CLUS, you do not have to use `forest` explicitly:

- if you are using, e.g., `MLCEnsemble`, `forest` is added automatically.
The calls below are equivalent:
    - `MLCEnsemble(..., forest=[], ssl=[])`,
    - `MLCEnsemble(..., ssl=[])`.
- if you are using, e.g., `MLCTree`, you will face an exception if you use `forest`.


### Java parameters

JVM parameters are passed to the constructor via the argument `java_parameters`, e.g.,

`java_parameters='-Xmx12G -Xms4G'`

The value of the argument is passed to the call `java <java params> clus.jar <clus params>` verbatim.

## Fit & Predict

### Fitting
When fitting, a temporary folder with

- training arff (where `xs_train` and `y_train` are dumped to)
- settings file (with the specified parameters)

is created.

Prior to fitting (calling `java <params1> clus.jar <params2>` internally),
the settings file is updated, so that CLUS outputs model files.

After the fitting, model files are loaded to your `model` object for later use.
Then, temporary folder is deleted.

### Predicting

When predicting, a temporary folder with

- testing arff (where `xs_test` and dummy target values are dumped to),
- settings file (with the specified parameters),
- model file(s)

is created. After predicting (again calling `java <params1> clus.jar <params2>` internally)
the prediction file is read. After that, the folder is deleted.

The predictions are of type `Dict[str, List[List[Any]]`. The keys are model names
(e.g., `Original` or `Forest with 4 trees`), wheres the values are lists of predictions for every
example in the test set.

Examples:

- regression (1 target): `{'Original': [[1.1], [2.1], ...], ...}`
- regression (2 targets): `{'Original': [[1.1, 1.2], [2.1, 2.2], ...], ...}`
- classification (1 target): `{'Original': [[('a', 0.87)], [('b', 0.51)], ...], ...}`
    - in addition to the predicted class value, the confidence is also given (e.g., model `Original` predicts
     `b` for the value of the first (and only target) for second example with confidence `0.51`).
     Confidence is not strictly defined in CLUS (might be probability, might be some other measure of support of the
     decision).
- classification (2 targets): `{'Original': [[('a', 0.87), ('x', 0.90)], [('b', 0.51), ('y', 0.61)], ...], ...}`
- MLC: same as classification, but the class values of every target are limited to '0' and '1'
- HMLC: `{'Original': [[('lab1', 0.87), ('lab2', 0.01), ...], [('lab1', 0.12), ('lab2', 0.61), ...], ...], ...}`
    - In contrast to MLC where the confidence always corresponds to the class value next to it,
    HMLC gives a confidence for example having a given label (thus, it is not weird if the confidence is less than `1/2`).

### Evaluation

PyClus does not provide the error measures. Use scikit instead, it has some great methods in `sklearn.metrics`.
