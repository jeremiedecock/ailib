# Machine Learning datasets

## Iris dataset

### Description

A classic toy classification dataset.

- 3 classes: ['setosa', 'versicolor', 'virginica']
- 50 samples per class
- 150 samples total
- 4 dimensions: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
- Features: real, positive (cm)

See:
- https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/descr/iris.rst
- http://scikit-learn.org/stable/datasets/index.html#datasets
- https://en.wikipedia.org/wiki/Iris_flower_data_set

### Source

Taken form scikit-learn (BSD license):

```
import json
from sklearn import datasets

iris = datasets.load_iris()

dataset = {"feature_names": iris.feature_names, "data": iris.data.tolist(), "target_names": iris.target_names.tolist(), "target": iris.target.tolist()}

with open("iris.json", "w") as fd:
    json.dump(dataset, fd, sort_keys=True, indent=4)
```

## Digits dataset

### Description

A classic toy classification dataset.

- 10 classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- 1797 samples
- ~180 samples per classes
- 64 dimensions (64 pixels in a 8x8 image of integer pixels in the range 0..16)
- Features: integer in the range 0..16

See:
- https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/descr/digits.rst
- http://scikit-learn.org/stable/datasets/index.html#datasets

### Source

Taken form scikit-learn (BSD license):

```
import json
from sklearn import datasets

digits = datasets.load_digits()

dataset = {"feature_names": None, "data": digits.data.tolist(), "target_names": digits.target_names.tolist(), "target": digits.target.tolist()}

with open("digits.json", "w") as fd:
    json.dump(dataset, fd, sort_keys=True, indent=4)
```

