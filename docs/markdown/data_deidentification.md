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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/data_deidentification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="UTRxpSlaczHY" -->
# Data De-Identification

In this deep dive, we will walk through some of the more advanced features to de-identify data with the Transform API, including bucketing, date shifts, masking, and entity replacements.

For this tutorial, we’ll use some sample [customer-like data](https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/customer-orders.csv) that contains a variety of interesting information that may need to be transformed depending on a downstream use case.

Transforms are highly declarative. Please take a look through our [Model Configuration](https://docs.gretel.ai/model-configurations) documentation to see all of the options for creating Policies and Rules.

<!-- #endregion -->

```python id="VEM6kjRsczHd"
%%capture

!pip install pyyaml Faker pandas
!pip install -U gretel-client
```

```python id="ZQ-TmAdwczHd"
# Specify your Gretel API key

import pandas as pd
from gretel_client import configure_session

pd.set_option("max_colwidth", None)

configure_session(api_key="prompt", cache="yes", validate=True)

```

```python
# Create our configuration with our Transforms Policies and Rules.
config = """# This example transform configuration supports the following dataset:
# https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/customer-orders.csv

schema_version: "1.0"
name: "example-transforms"

models:
  - transforms:
      data_source: "__tmp__"
      policies:
        - name: fake_identifiers
          rules:
            - name: fake_identifiers
              conditions:
                value_label:
                  - email_address
                  - phone_number
                  - ip_address
              transforms:
                - type: fake
                - type: hash # if a fake cannot be created
            - name: redact_names_locations
              conditions:
                field_label:
                  - person_name
                  - location
              transforms:
                - type: redact_with_char
            - name: dateshifter
              conditions:
                field_label:
                  - date
                  - datetime
              transforms:
                - type: dateshift
                  attrs:
                    min: 20
                    max: 20
                    formats: "%Y-%m-%d"
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
import yaml

from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll

# Create a project and model configuration.
project = create_or_get_unique_project(name="de-identify-transform")

model = project.create_model_obj(
    model_config=yaml.safe_load(config),
    data_source="https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/customer-orders.csv",
)

# Upload the training data.  Train the model.
model.submit_cloud()

poll(model)

```

```python
# Use the model to generate synthetic data.
record_handler = model.create_record_handler_obj(
    data_source="https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/customer-orders.csv",
)

record_handler.submit_cloud()

poll(record_handler)

# Compare results.  Here is our "before."
input_df = pd.read_csv(
    "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/customer-orders.csv"
)
print("input data, before de-identification")
print(input_df.head())

# And here is our "after."
deidentified = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")
print("input data, after de-identification")
deidentified.head()

```
