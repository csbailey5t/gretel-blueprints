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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/local_transform.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

## Label and Transform content locally

This notebook walks through training a transformation model and redacting PII locally in your environment.

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

# Simple policy to redact PII types with a character.
# Dates are shifted +/- 20 days based on the CustomerID field.
# Income is bucketized to 5000 number increments.

config = """
schema_version: 1.0
models:
  - transforms:
      data_source: "_"
      policies:
        - name: remove_pii
          rules:
            - name: fake_or_redact_pii
              conditions:
                value_label:
                  - person_name
                  - phone_number
                  - gender
                  - birth_date
              transforms:
                - type: redact_with_char
                  attrs:
                    char: X
            - name: dateshifter
              conditions:
                field_label:
                  - date
                  - datetime
                  - birth_date
              transforms:
                - type: dateshift
                  attrs:
                    min: 20
                    max: 20
                    formats: "%Y-%m-%d"
                    field_name: "CustomerID"        
            - name: bucketize-income
              conditions:
                field_name:
                  - YearlyIncome
              transforms:
                - type: numberbucket
                  attrs:
                    min: 0
                    max: 1000000
                    nearest: 5000
"""

```

```python
# Load and preview the DataFrame to train the transform model on.

df = pd.read_csv(data_source, nrows=500)
df.to_csv("training_data.csv", index=False)
df.head(5)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 582} id="xq2zj-6h_zh5" outputId="0587ddc8-ccb6-455b-f961-9392b4736d69"
project = create_or_get_unique_project(name="local-transform")
```

```python id="nvOhfvS4_zh5"
# The following cell will create the transform model and
# run a sample of the data set through the model. this sample
# can be used to ensure the model is functioning correctly
# before continuing.
transform = project.create_model_obj(
    model_config=yaml.safe_load(config), data_source="training_data.csv"
)

run = submit_docker_local(transform, output_dir="tmp/")

```

```python id="EAZLMwmG_zh6"
# Review the sampled classification report
# to get an overview of detected data types
report = json.loads(open("tmp/report_json.json.gz").read())
pd.DataFrame(report["metadata"]["fields"])

```

```python id="hL0COKZo_zh6"
# Next let's transform the remaining records using the transformation
# policy and model that was just created.
transform_records = transform.create_record_handler_obj(data_source="training_data.csv")

run = submit_docker_local(
    transform_records, model_path="tmp/model.tar.gz", output_dir="tmp/"
)

```

```python id="eVPQySOg_zh6"
# View the transformation report
report = json.loads(open("tmp/report_json.json.gz").read())
pd.DataFrame(report["metadata"]["fields"])

```

```python
# View the transformed data
results = pd.read_csv("tmp/data.gz")
results.head(5)

```

```python

```
