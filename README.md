# MDSVM

## What is MDSVM?

### Minimal Dependency

MDSVM is an implementation of Support Vector Machines with minimal dependency to other software.
Since it requires only [NumPy](http://www.numpy.org), it can work on various environments, for example [Google App Engine](https://cloud.google.com/appengine).

## Basic Usage

The usage is almost same as scikit-learn.

```python
import numpy as np
from mdsvm import csvc

num_p = 100
num_n = 100
dim = 2
x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
x_n = np.random.multivariate_normal(np.ones(dim) * 2, np.eye(dim), num_n)
x = np.vstack([x_p, x_n])
y = np.array([1.] * num_p + [-1.] * num_n)

# Hyper parameters
cost = 1e0
gamma = 0.1
max_iter = 2500
tol = 1e-5

clf_mdsvm = csvc.SVC(C=cost, kernel='rbf', max_iter=max_iter, gamma=gamma, tol=tol)

clf_mdsvm.fit(x, y)
print "Training Accuracy: {}".format(clf_mdsvm.score(x, y))
```

## Test

```
python -m unittest tests.test_csvc
```

## Benchmark

```
python -m bench.bench_csvc
```

## License

MIT