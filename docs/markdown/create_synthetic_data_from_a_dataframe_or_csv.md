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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/create_synthetic_data_from_a_dataframe_or_csv.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="UTRxpSlaczHY" -->
# Create synthetic data with the Python SDK

This notebook will walk you through the process of creating your own synthetic data using Gretel's Python SDK from a CSV or a DataFrame of your choosing.

To run this notebook, you will need an API key from the Gretel console, at https://console.gretel.cloud.

<!-- #endregion -->

```python id="VEM6kjRsczHd"
%%capture
!pip install -U gretel-client
```

```python id="ZQ-TmAdwczHd"
# Specify your Gretel API key

import pandas as pd
from gretel_client import configure_session

pd.set_option("max_colwidth", None)

configure_session(api_key="prompt", cache="yes", validate=True)

```

```python id="fmHDICI1oPS5"
# Create a project

from gretel_client.projects import create_or_get_unique_project

project = create_or_get_unique_project(name="synthetic-data")

```

<!-- #region id="4PD5B0U06ALs" -->
## Create the synthetic data configuration

Load the default configuration template. This template will work well for most datasets. View other templates at https://github.com/gretelai/gretel-blueprints/tree/main/config_templates/gretel/synthetics

<!-- #endregion -->

```python id="uIu3hkzoCzGz"
import json

from gretel_client.projects.models import read_model_config

config = read_model_config("synthetics/default")

# Set the model epochs to 50
config["models"][0]["synthetics"]["params"]["epochs"] = 50

print(json.dumps(config, indent=2))

```

<!-- #region id="s9LTh7GO6VIu" -->
## Load and preview the source dataset

Specify a data source to train the model on. This can be a local file, web location, or HDFS file.

<!-- #endregion -->

```python id="YMg9nX6SczHe"
# Load and preview the DataFrame to train the synthetic model on.
import pandas as pd

dataset_path = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"
df = pd.read_csv(dataset_path)
df.to_csv("training_data.csv", index=False)
df

```

<!-- #region id="WxnH8th-65Dh" -->
## Train the synthetic model

In this step, we will task the worker running in the Gretel cloud, or locally, to train a synthetic model on the source dataset.

<!-- #endregion -->

```python id="O4-E_F0qczHe"
from gretel_client.helpers import poll

model = project.create_model_obj(model_config=config, data_source="training_data.csv")
model.submit_cloud()

poll(model)

```

```python id="sPM-gaU6czHf"
# View the synthetic data

synthetic_df = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")

synthetic_df

```

<!-- #region id="69XYfU9k7fq4" -->
# View the synthetic data quality report

<!-- #endregion -->

```python id="zX8qsizqczHg" jupyter={"outputs_hidden": true} tags=[]
# Generate report that shows the statistical performance between the training and synthetic data

import IPython
from smart_open import open

IPython.display.HTML(data=open(model.get_artifact_link("report")).read(), metadata=dict(isolated=True))

```

<!-- #region id="6IkWOnVQ7oo1" -->
# Generate unlimited synthetic data

You can now use the trained synthetic model to generate as much synthetic data as you like.

<!-- #endregion -->

```python id="X0bI0OpI6W3Y"
# Generate more records from the model

record_handler = model.create_record_handler_obj(
    params={"num_records": 100, "max_invalid": 500}
)
record_handler.submit_cloud()
poll(record_handler)

```

```python id="uUIErjQ7CzGy"
synthetic_df = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")

synthetic_df

```
