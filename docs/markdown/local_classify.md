---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/local_classify.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

## Classify and label content locally

This notebook walks through training a classification model and labeling PII locally in your environment.

Follow the instructions here to set up your local environment: https://docs.gretel.ai/environment-setup

Prerequisites:

- Python 3.9+ (`python --version`).
- Ensure that Docker is running (`docker info`).
- The Gretel client SDK is installed and configured (`pip install -U gretel-client; gretel configure`).


```python id="ZLAlOI5f_zh2"
import json

import yaml
from smart_open import open
import pandas as pd

from gretel_client import submit_docker_local
from gretel_client.projects import create_or_get_unique_project

data_source = "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/example-datasets/bike-customer-orders.csv"

# Policy to search for sensitive data
# including a custom regular expression based search
config = """
schema_version: 1.0
models:
  - classify:
      data_source: "_"
      labels:
        - person_name
        - location
        - phone_number
        - date_time
        - birthdate
        - gender
        - acme/*
        
label_predictors:
  namespace: acme
  regex:
    user_id:
      patterns:
        - score: high
          regex: ^user_[\d]{5}$
"""

```

```python
# Load and preview the DataFrame to train the classification model on.

df = pd.read_csv(data_source, nrows=500)
df.to_csv("training_data.csv", index=False)
df

```

```python colab={"base_uri": "https://localhost:8080/", "height": 582} id="xq2zj-6h_zh5" outputId="0587ddc8-ccb6-455b-f961-9392b4736d69"
project = create_or_get_unique_project(name="local-classify")
```

```python id="nvOhfvS4_zh5"
# the following cell will create the classification model and
# run a sample of the data set through the model. this sample
# can be used to ensure the model is functioning correctly
# before continuing.
classify = project.create_model_obj(
    model_config=yaml.safe_load(config), data_source="training_data.csv"
)

run = submit_docker_local(classify, output_dir="tmp/")

```

```python id="EAZLMwmG_zh6"
# review the sampled classification report
report = json.loads(open("tmp/report_json.json.gz").read())
pd.DataFrame(report["metadata"]["fields"])

```

```python id="hL0COKZo_zh6"
# next let's classify the remaining records using the model
# that was just created.
classify_records = classify.create_record_handler_obj(data_source="training_data.csv")

run = submit_docker_local(
    classify_records, model_path="tmp/model.tar.gz", output_dir="tmp/"
)

```

```python id="eVPQySOg_zh6"
report = json.loads(open("tmp/report_json.json.gz").read())
pd.DataFrame(report["metadata"]["fields"])

```

```python
# Load results
results = pd.read_json("tmp/data.gz", lines=True)

# Examine labels found in the first record
results.iloc[0].to_dict()

```
