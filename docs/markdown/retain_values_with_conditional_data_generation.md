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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/retain_values_with_conditional_data_generation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="UTRxpSlaczHY" -->
# Retaining primary keys and field values with conditional data generation

Gretel supports a feature known as model conditioning (seeding) that will generate rows based on partial values from your training data. This is useful when you want to manually specify certain field values in the synthetic data, and let Gretel synthesize the rest of the row for you.

Use Cases for conditional data generation with Gretel:

- Create synthetic data that has the same number of rows as the training data
- You want to preserve some of the original row data (primary keys, dates, important categorical data).

When using conditional generation with Gretel's "seed" task, the model will generate one sample for each row of the seed dataframe, sorted in the same order.

In the example below, we'll use a combination of a primary key `client_id` and categorical fields `age` and `gender` as conditional inputs to the synthetic model, generating a new dataframe with the same primary key and categorical fields, but with the rest of the dataframe containing synthetically generated values.

<!-- #endregion -->

```python id="VEM6kjRsczHd"
%%capture

!pip install pyyaml smart_open pandas
!pip install -U gretel-client
```

```python id="ZQ-TmAdwczHd"
# Specify your Gretel API key

import pandas as pd
from gretel_client import configure_session

pd.set_option("max_colwidth", None)

configure_session(api_key="prompt", cache="yes", validate=True)

```

```python id="YMg9nX6SczHe"
# Load and preview dataset

import pandas as pd

dataset_path = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/customer_finance_data.csv"

# We will pull down the training data to drop an ID column.  This will help give us a better model.
training_df = pd.read_csv(dataset_path)

try:
    training_df.drop("disp_id", axis="columns", inplace=True)
except KeyError:
    pass  # incase we already dropped it

training_df

```

```python id="tvKsT56cjOFO"
from gretel_client.projects.models import  read_model_config
from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll


# Create a project and model configuration.
project = create_or_get_unique_project(name="conditional-data-example")

# Pull down the default synthetic config.  We will modify it slightly.
config = read_model_config("synthetics/default")

# Here we prepare an object to specify the conditional data generation task.
# In this example, we will retain the values for the seed fields below,
# use their values as inputs to the synthetic model.
fields = ["client_id", "age", "gender"]
task = {"type": "seed", "attrs": {"fields": fields}}
config["models"][0]["synthetics"]["task"] = task
config["models"][0]["synthetics"]["generate"] = {"num_records": len(training_df)}


# Fit the model on the training set
training_df.to_csv("train.csv", index=False)
model = project.create_model_obj(model_config=config, data_source="train.csv")

model.submit_cloud()

poll(model)

synthetic = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic.head()

```

```python id="He82umP5jOFP"
# Generate report that shows the statistical performance between the training and synthetic data

import IPython
from smart_open import open

IPython.display.HTML(data=open(model.get_artifact_link("report")).read(), metadata=dict(isolated=True))

```

```python id="VJMSsKsJj52c"
# Use the model to generate additional synthetic data.

seeds = training_df[fields]
seeds.to_csv("seeds.csv", index=False)

rh = model.create_record_handler_obj(
    data_source="seeds.csv", params={"num_records": len(seeds)}
)
rh.submit_cloud()

poll(rh)

synthetic_next = pd.read_csv(rh.get_artifact_link("data"), compression="gzip")
synthetic_next

```
