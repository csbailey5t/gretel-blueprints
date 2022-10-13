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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/time_series_generation_poc.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="Xbv1HhS1dXQq" -->
# Time Series Proof of of Concept

This blueprint demonstrates a full proof of concept for creating a synthetic financial time-series dataset and evaluating its privacy and accuracy for a predictive task

<!-- #endregion -->

```python id="QXBi_RW5dXQs"
%%capture

!pip install -U gretel-client
!pip install numpy pandas statsmodels matplotlib seaborn
!pip install -U scikit-learn

```

```python colab={"base_uri": "https://localhost:8080/"} id="W3eKIatM1mo4" outputId="56320388-d8b7-405f-f8c0-b8e5d1c4742e"
import pandas as pd
import numpy as np
import statsmodels as sm
from statsmodels.tsa.statespace import sarimax
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time

from typing import List, Dict
from gretel_client import configure_session

```

```python colab={"base_uri": "https://localhost:8080/"} id="-Kyza7XJdXQt" outputId="b87e0d03-9120-4aaf-ad21-dd211a960cca"
# Specify your Gretel API key

pd.set_option("max_colwidth", None)

configure_session(api_key="prompt", cache="yes", validate=True)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="73ciMrkldXQu" outputId="e0d39781-e93d-4d08-fa88-139e70e4b662"
# Load timeseries example to a dataframe

data_source = "https://gretel-public-website.s3.amazonaws.com/datasets/credit-timeseries-dataset.csv"
original_df = pd.read_csv(data_source)
original_df.to_csv("original.csv", index=False)
original_df

```

```python id="thFAMQaDuE8X"
# Gretel Transforms Configuration
config = """
schema_version: "1.0"
models:
    - transforms:
        data_source: "__tmp__"
        policies:
            - name: shiftnumbers
              rules:
                - name: shiftnumbers
                  conditions:
                    field_name:
                        - account_balance
                        - credit_amt
                        - debit_amt
                        - net_amt
                  transforms:
                    - type: numbershift
                      attrs:
                        min: 1
                        max: 100
                        field_name:
                            - date
                            - district_id
"""

```

```python colab={"base_uri": "https://localhost:8080/"} id="GgPpzZP9uKPx" outputId="e2bfa43f-4a6a-4f91-ccc5-6604007a1eea"
# De-identify the original dataset using the policy above
import yaml

from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll

# Create a project and model configuration.
project = create_or_get_unique_project(name="numbershift-transform")

model = project.create_model_obj(
    model_config=yaml.safe_load(config), data_source=data_source
)

# Upload the training data.  Train the model.
model.submit_cloud()
poll(model)

record_handler = model.create_record_handler_obj(data_source=data_source)
record_handler.submit_cloud()
poll(record_handler)

deid_df = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")

```

```python colab={"base_uri": "https://localhost:8080/", "height": 386} id="xFtDkVV_yYjU" outputId="21dfaa6b-899c-4d0a-cbbb-2ca8585716b4"
# View the transformation report

import json
from smart_open import open

report = json.loads(open(model.get_artifact_link("report_json")).read())
pd.DataFrame(report["metadata"]["fields"])

```

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="VnCiJT43wc1p" outputId="a1d2fad7-563a-4df3-cec4-92fa937dd14c"
# Here we sort and remove "net_amt" as it's a derived column,
# We will add back in after the data is synthesized
train_df = deid_df.copy()

train_df.sort_values("date", inplace=True)
train_cols = list(train_df.columns)
train_cols.remove("net_amt")
train_df = train_df.filter(train_cols)

# Here we noticed that some number have extremely long precision,
# so we round the data
train_df = train_df.round(1)
train_df.to_csv("train.csv", index=False)
train_df

```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="3tBtsrRQiawq" outputId="52121882-aa72-41a1-d34b-62f3cb71147b"
from gretel_client.projects.models import read_model_config

# Create a project and model configuration.
project = create_or_get_unique_project(name="ts-5544-regular-seed")

# Pull down the default synthetic config.  We will modify it slightly.
config = read_model_config("synthetics/default")

# Set up the seed fields
seed_fields = ["date", "district_id"]

task = {
    "type": "seed",
    "attrs": {
        "fields": seed_fields,
    },
}

# Fine tune model parameters. These are the parameters we found to work best.  This is "Run 20" in the document
config["models"][0]["synthetics"]["task"] = task

config["models"][0]["synthetics"]["params"]["vocab_size"] = 20
config["models"][0]["synthetics"]["params"]["learning_rate"] = 0.005
config["models"][0]["synthetics"]["params"]["epochs"] = 100
config["models"][0]["synthetics"]["params"]["gen_temp"] = 0.8
config["models"][0]["synthetics"]["params"]["reset_states"] = True
config["models"][0]["synthetics"]["params"]["dropout_rate"] = 0.5
config["models"][0]["synthetics"]["params"]["gen_temp"] = 0.8
config["models"][0]["synthetics"]["params"]["early_stopping"] = True
config["models"][0]["synthetics"]["privacy_filters"]["similarity"] = None
config["models"][0]["synthetics"]["privacy_filters"]["outliers"] = None
config["models"][0]["synthetics"]["generate"]["num_records"] = train_df.shape[0]

# Get a csv to work with, just dump out the train_df.
deid_df.to_csv("train.csv", index=False)

# Initiate a new model with the chosen config
model = project.create_model_obj(model_config=config, data_source="train.csv")

# Upload the training data.  Train the model.
model.submit_cloud()
poll(model)

synthetic = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic

```

```python id="4GyCx1wuyB0n"
# Add back in the derived column "net_amt"
net_amt = synthetic["credit_amt"] - synthetic["debit_amt"]
synthetic["net_amt"] = net_amt

# Save off the new synthetic data
synthetic.to_csv("synthetic.csv", index=False, header=True)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="JS0hXcJ-Y7Oo" outputId="1ea4100d-99c7-4164-dfa5-27d2394e8c53"
# View the Synthetic Performance Report
import IPython
from smart_open import open

IPython.display.HTML(data=open(model.get_artifact_link("report")).read(), metadata=dict(isolated=True))

```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="gi4d2NuuKQGV" outputId="547d3129-676c-4bda-8e44-aea19f38453b"
import matplotlib
import matplotlib.pyplot as plt


def plot_district_averages(
    synthetic: pd.DataFrame, training: pd.DataFrame, district_id: int
) -> pd.DataFrame:

    synthetic_data = synthetic.loc[synthetic["district_id"] == district_id]
    synthetic_data = synthetic_data.set_index("date")

    training_data = training.loc[training["district_id"] == district_id]
    training_data = training_data.set_index("date")

    combined = synthetic_data.join(
        training_data, lsuffix="_synthetic", rsuffix="_original"
    )
    plt.suptitle("District #" + str(district_id))

    for col in ["credit_amt", "debit_amt", "account_balance"]:
        fig = combined.plot(y=[f"{col}_synthetic", f"{col}_original"], figsize=(12, 8))
        plt.title("Time Series for District #" + str(district_id))

    return combined


combined = plot_district_averages(synthetic, train_df, 13)

```

```python id="24fAgRdLomsn"
import warnings

warnings.filterwarnings("ignore")


def ARIMA_run(
    data_paths: List[str],
    targets: List[str] = None,
    entity_column: str = "district_id",
    entities: List = None,
    date_column: str = "date",
    date_threshold: str = None,
) -> Dict[str, List[float]]:
    """
    Purpose of this function is to automate the run and scoring of SARIMAX models, so we can benchmark results against various different synthetic data configurations.
    The data paths from s3 are passed in, and then entire run, from loading in and sorting the data to creating a model and scoring it, is done via this function.
    The outputs are the target scores for each variable on each dataset's model. This gets used to create bar charts of the RMSE.
    With some fine tuning, this function can be made as a general purpose SARIMAX benchmark function for a variety of datasets.

    Args:
      data_paths: a list of paths to the data you want to create models and score with. These can be either local paths or ones from public buckets.
      targets: Which columns in the data will be your target variables?
      entity_column: This is purely used for datasets that have multiple time series data points from multiple places. Since this function was built with that in mind, it assumes that you will
      give a column that denotes those different places/entities. If None is provided, no handler has been built yet that can handle that.
      entities: This should be a list of the set of entities within the entity column.
      date_column: This should be something we can use to sort the data, so that the time series is read appropriately.
      date_threshold: This is to split the data into train and test. Whatever date you want to threshold by to make the train and test should be specified here.

    Outputs:
      target_scores: This will be a dictionary of RMSE scores for each target variable on each synthetic dataset.
    """
    target_scores = {}
    for target in targets:
        target_scores[target] = []
    for path in data_paths:
        sorted_data = pd.read_csv(path)
        sorted_data.sort_values(date_column, inplace=True)
        sorted_data.drop_duplicates(subset=[date_column, entity_column], inplace=True)

        print("Path: {}".format(path))
        for entity in entities:
            print("Entity: {}".format(entity))
            for target in targets:
                train_data = sorted_data[sorted_data[entity_column] == entity][
                    sorted_data[date_column] < date_threshold
                ]
                test_data = sorted_data[sorted_data[entity_column] == entity][
                    sorted_data[date_column] >= date_threshold
                ]

                model = sarimax.SARIMAX(
                    train_data[target], order=(0, 1, 1), seasonal_order=(1, 1, 0, 12)
                )
                res = model.fit()

                preds = res.forecast(len(test_data[target]))
                rmse = mean_squared_error(test_data[target], preds, squared=False)
                target_scores[target].append(rmse)
                print("Target: {}".format(target))
                print("RMSE: {}".format(rmse))

    return target_scores

```

```python colab={"base_uri": "https://localhost:8080/"} id="WsK5p3YB204I" outputId="3fbc4baa-6599-4c0f-e5ce-45bcfcecd00e"
target_scores = ARIMA_run(
    ["synthetic.csv", "original.csv"],
    targets=["net_amt", "account_balance", "credit_amt", "debit_amt"],
    entities=[13],
    date_threshold="1998-01-01",
)
target_scores

```

```python colab={"base_uri": "https://localhost:8080/", "height": 352} id="d9BewEn_3B6x" outputId="7d6946ae-0825-4c4c-abd9-c98106ae1c80"
import plotly.express as px

results = pd.DataFrame.from_dict(target_scores)
results["method"] = ["synthetic", "real world"]
results.plot.bar(x="method", title="RMSE per field and run in synthetic timeseries")

```
