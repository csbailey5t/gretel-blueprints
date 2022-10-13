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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/synthetic_data_walkthrough.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="UTRxpSlaczHY" -->
# Create synthetic data with the Python SDK

This notebook utilizes Gretel's SDK and APIs to create a synthetic version of a popular machine learning financial dataset.

To run this notebook, you will need an API key from the Gretel console, at https://console.gretel.cloud.

<!-- #endregion -->

```python id="VEM6kjRsczHd"
%%capture
!pip install pyyaml smart_open pandas
!pip install -U gretel-client
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZQ-TmAdwczHd" outputId="03aa9c40-01f8-4711-a80b-52322721ee4c"
# Specify your Gretel API key

import pandas as pd
from gretel_client import configure_session

pd.set_option("max_colwidth", None)

configure_session(api_key="prompt", cache="yes", validate=True)

```

```python id="fmHDICI1oPS5"
# Create a project

from gretel_client.projects import create_or_get_unique_project

project = create_or_get_unique_project(name="walkthrough-synthetic")

```

<!-- #region id="4PD5B0U06ALs" -->
## Create the synthetic data configuration

Load the default configuration template. This template will work well for most datasets. View other templates at https://github.com/gretelai/gretel-blueprints/tree/main/config_templates/gretel/synthetics

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="uIu3hkzoCzGz" outputId="94c32679-4a9c-4af3-95d2-1fbda2e617ed"
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

```python colab={"base_uri": "https://localhost:8080/", "height": 571} id="YMg9nX6SczHe" outputId="18d0a1f8-07cd-4811-a385-9c159a58b26a"
# Load and preview dataset to train the synthetic model on.
import pandas as pd

model = project.create_model_obj(
    model_config=config,
    data_source="https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv",
)

pd.read_csv(model.data_source)

```

<!-- #region id="WxnH8th-65Dh" -->
## Train the synthetic model

In this step, we will task the worker running in the Gretel cloud, or locally, to train a synthetic model on the source dataset.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="O4-E_F0qczHe" outputId="6b82092d-ded1-43f0-f1ac-115dd8992956"
from gretel_client.helpers import poll

model.submit_cloud()

poll(model)

```

<!-- #region id="2bgWKArX7QGf" -->
# View the generated synthetic data

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 538} id="sPM-gaU6czHf" outputId="e29e9b29-06f2-40a3-de4d-5f9e6d41b621"
# View the synthetic data

synthetic_df = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")

synthetic_df.head()

```

<!-- #region id="69XYfU9k7fq4" -->
# View the synthetic data quality report

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="zX8qsizqczHg" jupyter={"outputs_hidden": true} outputId="2daf44a8-13f5-4e2c-cccc-b26a5a59d461" tags=[]
# Generate report that shows the statistical performance between the training and synthetic data

import IPython
from smart_open import open

IPython.display.HTML(data=open(model.get_artifact_link("report")).read(), metadata=dict(isolated=True))

```

<!-- #region id="6IkWOnVQ7oo1" -->
# Generate unlimited synthetic data

You can now use the trained synthetic model to generate as much synthetic data as you like.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="X0bI0OpI6W3Y" outputId="7faf358b-e3af-4e3f-8368-aeb940d19c42"
# Generate more records from the model

record_handler = model.create_record_handler_obj(
    params={"num_records": 100, "max_invalid": 500}
)

record_handler.submit_cloud()

poll(record_handler)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 554} id="uUIErjQ7CzGy" outputId="4d1518e2-ee5f-4f00-cab5-81c75b54e9ca"
synthetic_df = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")

synthetic_df.head()

```
