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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/redact_pii.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="UTRxpSlaczHY" -->
# Redact PII

In this blueprint, we will create a transform policy to identify and redact or replace PII with fake values. We will then use the SDK to transform a dataset and examine the results.

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
config = """schema_version: "1.0"
name: "Redact PII"
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
                  - credit_card_number
                  - phone_number
                  - us_social_security_number
                  - email_address
                  - custom/*
              transforms:
                - type: fake
                - type: redact_with_char
                  attrs:
                    char: X
label_predictors:
  namespace: custom
  regex:
    user_id:
      patterns:
        - score: high
          regex: 'user_[\d]{5}'
"""

```

```python
from faker import Faker

# Use Faker to make training and test data.
def fake_pii_csv(filename, lines=100):
    fake = Faker()
    with open(filename, "w") as f:
        f.write("id,name,email,phone,visa,ssn,user_id\n")
        for i in range(lines):
            _name = fake.name()
            _email = fake.email()
            _phone = fake.phone_number()
            _cc = fake.credit_card_number()
            _ssn = fake.ssn()
            _id = f'user_{fake.numerify(text="#####")}'
            f.write(f"{i},{_name},{_email},{_phone},{_cc},{_ssn},{_id}\n")


fake_pii_csv("train.csv")
fake_pii_csv("test.csv")

```

```python
import yaml

from gretel_client.projects import create_or_get_unique_project
from gretel_client.helpers import poll

# Create a project and model configuration.
project = create_or_get_unique_project(name="redact-pii-transform")

model = project.create_model_obj(
    model_config=yaml.safe_load(config), data_source="train.csv"
)

# Upload the training data.  Train the model.
model.submit_cloud()

poll(model)

```

```python
# Use the model to generate synthetic data.
record_handler = model.create_record_handler_obj(data_source="test.csv")

record_handler.submit_cloud()

poll(record_handler)

# Compare results.  Here is our "before."
train_df = pd.read_csv("test.csv")
print("test.csv head, before redaction")
print(train_df.head())

# And here is our "after."
transformed = pd.read_csv(record_handler.get_artifact_link("data"), compression="gzip")
print("test.csv head, after redaction")
transformed.head()

```
