---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/boost_massively_imbalanced_set.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

```python
%%capture
!pip install pyyaml numpy pandas sklearn smart_open xgboost
!pip install -U gretel-client
```

```python
# Specify your Gretel API key

import pandas as pd
from gretel_client import configure_session

pd.set_option("max_colwidth", None)

configure_session(api_key="prompt", cache="yes", validate=True)

```

```python
# Create imbalanced train and test data
# We will use sklearn's make_classification to create a test dataset.
# Or, load your own dataset as a Pandas DataFrame.

CLASS_COLUMN = "Class"  # the labeled classification column
CLASS_VALUE = 1  # the minority classification label to boost
MAX_NEIGHBORS = 5  # number of KNN neighbors to use per positive datapoint
SYNTHETIC_PERCENT = 10  # generate SYNTHETIC_PERCENT records vs. source data

# Create imbalanced test dataset
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


n_features = 15
n_recs = 10000


def create_dataset(n_features: int) -> pd.DataFrame:
    """Use sklearn to create a massively imbalanced dataset"""
    X, y = make_classification(
        n_samples=n_recs,
        n_features=n_features,
        n_informative=10,
        n_classes=2,
        weights=[0.95],
        flip_y=0.0,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"feature_{x}" for x in range(n_features)])
    df = df.round(6)
    df[CLASS_COLUMN] = y
    return df


dataset = create_dataset(n_features=n_features)
train, test = train_test_split(dataset, test_size=0.2)

train.head()

```

```python
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Split positive and negative datasets
positive = train[train[CLASS_COLUMN] == CLASS_VALUE]
print(f"Positive records shape (rows, columns): {positive.shape}")

# Train a nearest neighbor model on the negative dataset
neighbors = NearestNeighbors(n_neighbors=MAX_NEIGHBORS, algorithm="ball_tree")
neighbors.fit(train)

# Locate the nearest neighbors to the positive (minority) set,
# and add to the training set.
nn = neighbors.kneighbors(positive, MAX_NEIGHBORS, return_distance=False)
nn_idx = list(set([item for sublist in nn for item in sublist]))
nearest_neighbors = train.iloc[nn_idx, :]

oversample = pd.concat([positive] * 5)
training_set = pd.concat([oversample, nearest_neighbors]).sample(frac=1)

training_set.head()

```

```python
from smart_open import open
import yaml

from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll

# Create a project and model configuration.
project = create_or_get_unique_project(name="boost-imbalanced-synthetic")

# If you want to use a different config or modify it before creating the model,
# try something like this (yes, we have other stock configs in that repo)
#   from gretel_client.projects.models import read_model_config
#   config = read_model_config("synthetics/default")

# Get a csv to work with, just dump out the training_set.
training_set.to_csv("train.csv", index=False)

# Here we just use a shortcut to specify the default synthetics config.
# Yes, you can use other shortcuts to point at some of the other stock configs.
model = project.create_model_obj(
    model_config="synthetics/default", data_source="train.csv"
)


# Upload the training data.  Train the model.
model.submit_cloud()
poll(model)

recs_to_generate = int(len(dataset.values) * (SYNTHETIC_PERCENT / 100.0))

# Use the model to generate synthetic data.
record_handler = model.create_record_handler_obj(
    params={"num_records": recs_to_generate, "max_invalid": recs_to_generate}
)
record_handler.submit_cloud()

poll(record_handler)

synthetic_df = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")
synthetic = synthetic_df[
    synthetic_df[CLASS_COLUMN] == CLASS_VALUE
]  # Keep only positive examples

synthetic.head()

```

```python
df = pd.concat(
    [
        train.assign(Type="train"),
        test.assign(Type="test"),
        synthetic.assign(Type="synthetic"),
    ]
)
df.reset_index(inplace=True)
df.to_csv("combined-boosted-df.csv")
project.upload_artifact("combined-boosted-df.csv")

# Save to local CSV
synthetic.to_csv("boosted-synthetic.csv", index=False)
project.upload_artifact("boosted-synthetic.csv")

print(f"View this project at: {project.get_console_url()}")

```

```python
# Visualize distribution of positive and negative examples in our 
# normal vs. boosted datasets

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_distributions(test: pd.DataFrame, train: pd.DataFrame, synthetic: pd.DataFrame):
    """ Plot the distribution of positive (e.g. fraud) vs negative 
        e.g. (non-fraud) examples. 
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    fig = plt.figure(1, figsize=(12, 9))

    dataframes = {
        "test": test,
        "train": train,
        "boosted": pd.concat([train, synthetic])
    }

    idx = 0
    for name, df in dataframes.items():
        df.Class.value_counts().plot.bar(ax=axes[idx], title=name)
        idx+=1

visualize_distributions(test, train, synthetic)
```

```python
## Use PCA to visualize highly dimensional data

# We will label each data class as:
# * Training negative: 0
# * Training positive: 1
# * Synthetic positive: 2 (our synthetic data points used to boost training data)
# * Test positive: 3 (not cheating here, we already trained the classifier)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_visualization_dataframe(train: pd.DataFrame) -> pd.DataFrame:
    # Build a new visualization dataframe from our training data
    train_vis = train

    # Add in positive synthetic results
    train_vis = pd.merge(train, synthetic, indicator=True, how="outer")
    train_vis.loc[(train_vis._merge == "right_only"), "Class"] = 2
    train_vis = train_vis.drop(columns=["_merge"])

    # Add in positive results from the test set
    train_vis = pd.merge(
        train_vis, test[test["Class"] == 1], indicator=True, how="outer"
    )
    train_vis.loc[
        (train_vis._merge == "right_only") | (train_vis._merge == "both"), "Class"
    ] = 3
    train_vis = train_vis.drop(columns=["_merge"])
    return train_vis


def visualize_pca_2d(train_vis: pd.DataFrame):
    X = train_vis.iloc[:, :-1]
    y = train_vis["Class"]

    fig = plt.figure(1, figsize=(12, 9))
    plt.clf()
    plt.cla()

    pca = PCA(n_components=2)
    x_std = StandardScaler().fit_transform(X)
    projected = pca.fit_transform(x_std)

    labels = ["Train Negative", "Train Positive", "Synthetic Positive", "Test Positive"]
    size_map = {0: 25, 1: 50, 2: 75, 3: 50}
    sizes = [size_map[x] for x in y]

    scatter = plt.scatter(
        projected[:, 0], projected[:, 1], c=y, s=sizes, cmap=plt.cm.plasma, alpha=0.8
    )
    plt.title = f"PCA plot of {n_features}-dimension classification dataset"
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.show()


# Visualize PCA distribution in 2D
train_vis = create_visualization_dataframe(train)
visualize_pca_2d(train_vis)

```

```python
# Plot PCA scatter in 3 dimensions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn import datasets


def visualize_pca_3d(train_vis: pd.DataFrame):
    X = train_vis.iloc[:, :-1]
    y = train_vis["Class"]

    np.random.seed(5)

    fig = plt.figure(1, figsize=(12, 9))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    plt.cla()
    pca = decomposition.PCA(n_components=3)
    labels = ["Train Negative", "Train Positive", "Synthetic Positive", "Test Positive"]
    size_map = {0: 25, 1: 50, 2: 75, 3: 50}
    sizes = [size_map[x] for x in y]

    pca.fit(X)
    X = pca.transform(X)

    scatter = ax.scatter(
        X[:, 0], X[:, 1], X[:, 2], c=y, s=sizes, cmap=plt.cm.plasma, alpha=1.0
    )

    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.show()


# Visualize PCA distribution in 3D
visualize_pca_3d(train_vis)

```

```python
# Train an XGBoost model and compare accuracies on the original (normal)
# vs. augmented training data (train + synthetic) datasets.

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def train_classifier(name: str, train: pd.DataFrame, test: pd.DataFrame):
    """Train our predictor with XGBoost"""

    # Encode labels and categorical variables before training prediction model
    X_train = train.iloc[:, :-1]
    y_train = train["Class"]
    X_test = test.iloc[:, :-1]
    y_test = test["Class"]

    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    np.set_printoptions(precision=2)
    print("%s : XGBoost Model prediction accuracy: %.2f%%" % (name, accuracy * 100.0))
    return model, y_pred


# Train models on normal and augmented data
model_normal, y_pred = train_classifier("normal", train, test)
model_boosted, y_pred = train_classifier("boosted", pd.concat([train, synthetic]), test)

```

```python
# A confusion matrix gives better insight into per-class performance
# than overall model accuracy.

# As a thought experiment, consider creating a model to predict whether
# an account will submit an insurance claim. Our goal is to maximize
# accuracy at predicting the minority (positive) set, above those who
# will not submit a claim. Try to maximize the diagonal (TP) elements of the
# confusion matrix, particularly the bottom right.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def print_confusion_matrix(name: str, model: pd.DataFrame, test: pd.DataFrame):
    """Plot normalized and non-normalized confusion matrices"""
    print("")
    print("")
    print(f"Plotting confusion matrices for: {name} model")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig = plt.figure(1, figsize=(12, 9))
    X_test = test.iloc[:, :-1]
    y_test = test["Class"]

    titles_options = [
        (f"{name} : Confusion matrix, without normalization", None),
        (f"{name} : Normalized confusion matrix", "true"),
    ]

    idx = 0
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(
            model,
            X_test,
            y_test,
            display_labels=["Negative", "Positive"],
            cmap=plt.cm.Blues,
            normalize=normalize,
            ax=axes[idx],
        )
        disp.ax_.set_title(title)
        idx += 1

    plt.show()


print_confusion_matrix("normal", model_normal, test)
print_confusion_matrix("boosted", model_boosted, test)

```
