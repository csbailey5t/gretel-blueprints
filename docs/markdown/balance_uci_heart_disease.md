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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/balance_uci_heart_disease.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="BujHsjP2zY6m" -->
This notebook demonstrates using Gretel.ai's conditional sampling to balance the gender attributes in a popular healthcare dataset, resulting in both better ML model accuracy, and potentially a more ethically fair training set.

The Heart Disease dataset published by University of California Irvine is one of the top 5 datasets on the data science competition site Kaggle, with 9 data science tasks listed and 1,014+ notebook kernels created by data scientists. It is a series of health 14 attributes and is labeled with whether the patient had a heart disease or not, making it a great dataset for prediction.

<!-- #endregion -->

```python id="hbBXoBVyvkZ4"
%%capture
!pip install gretel_client xgboost
```

```python colab={"base_uri": "https://localhost:8080/"} id="PR_EA4Z-v8WM" outputId="89e66d2d-a793-4ba0-9c83-0ff8e67fe79e"
from gretel_client import configure_session

configure_session(api_key="prompt", cache="yes", validate=True)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 423} id="YMg9nX6SczHe" outputId="0be46d67-6f51-47f2-8ed3-ca380744c280"
# Load and preview dataset

import pandas as pd

# Create from Kaggle dataset using an 70/30% split.
train = pd.read_csv(
    "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/uci-heart-disease/heart_train.csv"
)
test = pd.read_csv(
    "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/uci-heart-disease/heart_test.csv"
)

train

```

```python colab={"base_uri": "https://localhost:8080/", "height": 560} id="BTeNPvgKvkZ6" outputId="d5c4c979-918c-4a48-d959-f8d47d937706"
# Plot distributions in real world data

pd.options.plotting.backend = "plotly"

df = train.sex.copy()
df = df.replace(0, "female").replace(1, "male")

print(
    f"We will need to augment training set with an additional {train.sex.value_counts()[1] - train.sex.value_counts()[0]} records to balance gender class"
)
df.value_counts().sort_values().plot(kind="barh", title="Real world distribution")

```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="tvKsT56cjOFO" outputId="b0ed60db-3f8d-419f-f32f-32b680164fdd"
# Train a synthetic model on the training set

from gretel_client import projects
from gretel_client.projects.models import read_model_config
from gretel_client.helpers import poll

# Create a project and model configuration.
project = projects.create_or_get_unique_project(name="uci-heart-disease")

config = read_model_config("synthetics/default")

# Here we prepare an object to specify the conditional data generation task.
fields = ["sex"]
task = {"type": "seed", "attrs": {"fields": fields}}
config["models"][0]["synthetics"]["task"] = task
config["models"][0]["synthetics"]["generate"] = {"num_records": 500}
config["models"][0]["synthetics"]["privacy_filters"] = {
    "similarity": None,
    "outliers": None,
}


# Fit the model on the training set
model = project.create_model_obj(model_config=config)
train.to_csv("train.csv", index=False)
model.data_source = "train.csv"
model.submit_cloud()

poll(model)

synthetic = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic

```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="VJMSsKsJj52c" outputId="9a29ff2f-660e-4569-d2d7-3130192581e4"
# Conditionaly sample records from the synthetic data model using `seeds`
# to augment the real world training data


num_rows = 5000
seeds = pd.DataFrame(index=range(num_rows), columns=["sex"]).fillna(0)
delta = train.sex.value_counts()[1] - train.sex.value_counts()[0]
seeds["sex"][int((num_rows + delta) / 2) :] = 1
seeds.sample(frac=1).to_csv("seeds.csv", index=False)

rh = model.create_record_handler_obj(
    data_source="seeds.csv", params={"num_records": len(seeds)}
)
rh.submit_cloud()

poll(rh)

synthetic = pd.read_csv(rh.get_artifact_link("data"), compression="gzip")
augmented = pd.concat([synthetic, train])
augmented

```

```python colab={"base_uri": "https://localhost:8080/", "height": 560} id="ZG3TEyfxvkZ8" outputId="8689cafd-019f-4880-bb0f-b260895af564"
# Plot distributions in the synthetic data


print(
    f"Augmented synthetic dataset with an additional {delta} records to balance gender class"
)
df = augmented.sex.copy()
df = df.replace(0, "female").replace(1, "male")
df.value_counts().sort_values().plot(
    kind="barh", title="Augmented dataset distribution"
)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 756} id="f-nDGh46vkZ8" outputId="5716d609-e1c4-46f5-9add-a8d6910ef556"
# Compare real world vs. synthetic accuracies using popular classifiers

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import plotly.express as px


def classification_accuracy(data_type, dataset, test) -> dict:

    accuracies = []
    x_cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
    y_col = "target"

    rf = RandomForestClassifier(n_estimators=1000, random_state=1)
    rf.fit(dataset[x_cols], dataset[y_col])
    acc = rf.score(test[x_cols], test[y_col]) * 100
    accuracies.append([data_type, "RandomForest", acc])
    print(" -- Random Forest: {:.2f}%".format(acc))

    svm = SVC(random_state=1)
    svm.fit(dataset[x_cols], dataset[y_col])
    acc = svm.score(test[x_cols], test[y_col]) * 100
    accuracies.append([data_type, "SVM", acc])
    print(" -- SVM: {:.2f}%".format(acc))

    knn = KNeighborsClassifier(n_neighbors=2)  # n_neighbors means k
    knn.fit(dataset[x_cols], dataset[y_col])
    acc = knn.score(test[x_cols], test[y_col]) * 100
    accuracies.append([data_type, "KNN", acc])
    print(" -- KNN: {:.2f}%".format(acc))

    dtc = DecisionTreeClassifier()
    dtc.fit(dataset[x_cols], dataset[y_col])
    acc = dtc.score(test[x_cols], test[y_col]) * 100
    accuracies.append([data_type, "DecisionTree", acc])
    print(" -- Decision Tree Test Accuracy {:.2f}%".format(acc))

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="error")
    xgb.fit(dataset[x_cols], dataset[y_col])
    acc = xgb.score(test[x_cols], test[y_col]) * 100
    accuracies.append([data_type, "XGBoost", acc])
    print(" -- XGBoostClassifier: {:.2f}%".format(acc))

    return accuracies


print("Calculating real world accuracies")
realworld_acc = classification_accuracy("real world", train, test)
print("Calculating synthetic accuracies")
synthetic_acc = classification_accuracy("synthetic", augmented, test)

comparison = pd.DataFrame(
    realworld_acc + synthetic_acc, columns=["data_type", "algorithm", "acc"]
)
colours = {
    "synthetic": "#3EC1CD",
    "synthetic1": "#FCB94D",
    "real world": "#9ee0e6",
    "real world1": "#fddba5",
}

fig = px.bar(
    comparison,
    x="algorithm",
    y="acc",
    color="data_type",
    color_discrete_map=colours,
    barmode="group",
    text_auto=".4s",
    title="Real World vs. Synthetic Data for <b>all classes</b>",
)
fig.update_layout(legend_title_text="<b>Real world v. Synthetic</b>")
fig.show()

```

```python colab={"base_uri": "https://localhost:8080/"} id="z8XG1abginmY" outputId="5d1ae12f-6cdc-45d7-9198-ef8abee12e46"
print("Calculating real world class accuracies")
realworld_male = classification_accuracy(
    "realworld_male", train, test.loc[test["sex"] == 1]
)
realworld_female = classification_accuracy(
    "realworld_female", train, test.loc[test["sex"] == 0]
)
print("Calculating synthetic class accuracies")
synthetic_male = classification_accuracy(
    "synthetic_male", augmented, test.loc[test["sex"] == 1]
)
synthetic_female = classification_accuracy(
    "synthetic_female", augmented, test.loc[test["sex"] == 0]
)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 542} id="5xky1T471Gec" outputId="7def9d19-34e4-4df4-e7c3-9dd9e9f6b8bb"
# Plot male (majority class) heart disease detection accuracies (real world vs. synthetic)
colours = {
    "synthetic_male": "#3EC1CD",
    "synthetic_female": "#FCB94D",
    "realworld_male": "#9ee0e6",
    "realworld_female": "#fddba5",
}

comparison = pd.DataFrame(
    realworld_male + synthetic_male + realworld_female + synthetic_female,
    columns=["data_type", "algorithm", "acc"],
)
fig = px.bar(
    comparison,
    x="algorithm",
    y="acc",
    color="data_type",
    color_discrete_map=colours,
    barmode="group",
    text_auto=".4s",
    title="Real World vs. Synthetic Accuracy for <b>Male and Female Heart Disease Detection</b>",
)
fig.update_layout(legend_title_text="<b>Real world v. Synthetic</b>")
fig.show()

```
