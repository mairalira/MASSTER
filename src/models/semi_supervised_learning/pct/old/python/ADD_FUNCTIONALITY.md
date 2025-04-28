# How to add a new model?

Create a class in `clus.models.clus_model_type`. Follow the patterns in

- `ClusModelTree`
- `ClusModelEnsemble`
- `ClusModelRelief`

Lets name it `ClusModelRules`.

After that, create classes such is `ClassificationRules`, `RegressionRules`, etc. in
`clus.models.classification`, `clus.models.regression`, etc.


`ClassificationRules` should have `ClusModelClassification` and `ClusModelRules` as parent classes:

- The first one takes care of classification related stuff (checks whether targets are nominal etc.)
- The second one takes care of the correct arguments (do you use `-rules`, add it if not, etc.)


Possibly, you will have to override `fit`, `predict` and maybe even `_read_predictions`.
