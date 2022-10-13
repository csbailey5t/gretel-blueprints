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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/create_synthetic_data_from_time_series.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="GUnwBU_zYz2D" -->
# Synthesize Time Series data from your own DataFrame

This Blueprint demonstrates how to create synthetic time series data with Gretel. We assume that within the dataset
there is at least:

1. A specific column holding time data points

2. One or more columns that contain measurements or numerical observations for each point in time.

For this Blueprint, we will generate a very simple sine wave as our time series data.

<!-- #endregion -->

```python id="b4-JFrb-Yz2G"
%%capture

!pip install numpy matplotlib pandas
!pip install -U gretel-client
```

```python id="pHShf3MdYz2I"
# Specify your Gretel API key

import pandas as pd
from gretel_client import configure_session

pd.set_option("max_colwidth", None)

configure_session(api_key="prompt", cache="yes", validate=True)

```

```python id="moLu6jA3Yz2I"
# Create a simple timeseries with a sine and cosine wave

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

day = 24 * 60 * 60
year = 365.2425 * day


def load_dataframe() -> pd.DataFrame:
    """Create a time series x sin wave dataframe."""
    df = pd.DataFrame(columns=["date", "sin", "cos", "const"])

    df.date = pd.date_range(start="2017-01-01", end="2021-07-01", freq="4h")
    df.sin = 1 + np.sin(df.date.astype("int64") // 1e9 * (2 * np.pi / year))
    df.sin = (df.sin * 100).round(2)

    df.cos = 1 + np.cos(df.date.astype("int64") // 1e9 * (2 * np.pi / year))
    df.cos = (df.cos * 100).round(2)

    df.date = df.date.apply(lambda d: d.strftime("%Y-%m-%d"))

    df.const = "abcxyz"

    return df


train_df = load_dataframe()
train_df.set_index("date").plot(figsize=(12, 8))

```

<!-- #region id="p7IlPWWPb38C" -->
# Fine-tuning hyper-parameters for time-series

In this cell, we define the `date` field as the time_field for our task, and `sin` and `cos` as trend fields where we wish to model the differences between each time step.

## Hyper parameters

- `vocab_size` is set to 0 to use character-based tokenization vs. sentencepiece
- `predict_batch_size` is set to 1, which reduces generation speed but maximimizes use of model to replay long-term dependencies from the training sequences
- `validation_split` is set to False, as randomly sampled time-series records will have an information leakage problem between the train and test sets.
- `learning_rate` is set to 0.001, which increases training time but gives the model additional time to learn.

<!-- #endregion -->

```python id="F1q3ighmYz2J"
from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll
from gretel_client.projects.models import read_model_config


# Create a project and model configuration.
project = create_or_get_unique_project(name="time-series-synthetic")

# Pull down the default synthetic config.  We will modify it slightly.
config = read_model_config("synthetics/default")


# Here we create an object to specify the timeseries task.
time_field = "date"
trend_fields = ["sin", "cos"]

task = {
    "type": "time_series",
    "attrs": {"time_field": time_field, "trend_fields": trend_fields},
}

config["models"][0]["synthetics"]["task"] = task
config["models"][0]["synthetics"]["params"]["epochs"] = 100
config["models"][0]["synthetics"]["params"]["vocab_size"] = 0
config["models"][0]["synthetics"]["params"]["learning_rate"] = 1e-3
config["models"][0]["synthetics"]["params"]["predict_batch_size"] = 1
config["models"][0]["synthetics"]["params"]["validation_split"] = False
config["models"][0]["synthetics"]["params"]["reset_states"] = True
config["models"][0]["synthetics"]["params"]["overwrite"] = True
config["models"][0]["synthetics"]["generate"]["num_records"] = train_df.shape[0]
config["models"][0]["synthetics"]["generate"]["max_invalid"] = train_df.shape[0]

# Get a csv to work with, just dump out the train_df.
train_df.to_csv("train.csv", index=False)

model = project.create_model_obj(model_config=config, data_source="train.csv")

# Upload the training data. Train the model.
model.submit_cloud()
poll(model)

synthetic = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic

```

```python id="DoT24lMpYz2K"
# Does the synthetic data look similar? Yep!
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
for k, v in enumerate(trend_fields):
    train_df[["date", v]].set_index("date").plot(ax=axs[k], ls="--")
    synthetic[["date", v]].set_index("date").plot(ax=axs[k], alpha=0.7)
    axs[k].legend(["training", "synthetic"], loc="lower right")
    axs[k].set_title(v)
plt.show()

```

```python id="zfe_3m68ajwn"
# For time series data we dump out the date column to seed the record handler.
train_df["date"].to_csv("date_seeds.csv", index=False)

# Use the model to generate more synthetic data.
record_handler = model.create_record_handler_obj(
    params={"num_records": 5000, "max_invalid": 5000},
    data_source="date_seeds.csv",
)

record_handler.submit_cloud()

poll(record_handler)

# Create a second synthetic dataframe
synthetic_2 = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")
synthetic_2

```

```python id="wZxrdBOdaxxk"
# Does the synthetic data look similar? Yep!
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
for k, v in enumerate(trend_fields):
    train_df[["date", v]].set_index("date").plot(ax=axs[k], ls="--")
    synthetic[["date", v]].set_index("date").plot(ax=axs[k], alpha=0.7)
    synthetic_2[["date", v]].set_index("date").plot(ax=axs[k], alpha=0.7)
    axs[k].legend(["training", "synthetic", "synthetic_2"], loc="lower right")
    axs[k].set_title(v)
plt.show()

```
