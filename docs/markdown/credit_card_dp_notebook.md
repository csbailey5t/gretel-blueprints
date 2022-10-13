---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: 'Python 3.8.13 (''tf'': conda)'
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/credit_card_dp_notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

# Differentially private synthetic model with credit card dataset

This blueprint implements a practical attack on a credit card dataset. We tune various parameters and privacy settings of a synthetic model to measure its ability in memorizing canaries inserted into dataset. We show that enabling differential privacy (DP) can provide greater protection from memorization of canaries.

```python
%%capture
%pip install gretel-client 
```

```python
# Specify the Gretel API key. You can acquire this from the Gretel Console 
# @ https://console.gretel.cloud

import pandas as pd

from gretel_client import configure_session
pd.set_option('max_colwidth', None)
configure_session(api_key="prompt", cache="yes", validate=True)
```

```python
# Load the credit card transaction fraud detection dataset to a dataframe.
data_source = "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/CreditCard_Fraud_Transactions.csv"
data = pd.read_csv(data_source , index_col = [0])
# See the raw dataset:
data.head()
```

```python
# Checking the ranges of the credit card number's length:
data["cc_num"].apply(lambda x:len(str(x))).value_counts().rename_axis("credit card number of digits").reset_index(name = "count")
```

```python
# Reduce the number of the records to 28K and select 4 feature columns:
import random
random.seed(42)

sample_data = data.filter(["cc_num","first","last","gender"],axis =1)
# Since there are various credit card digit counts, we use te last 4 digits which is mostly common.
sample_data["cc_num"] = sample_data["cc_num"].apply(lambda x:(str(x)[-4:]))
# Just Sampling 28K dataset:
sample_df = sample_data.sample(n = 28000,random_state = 62).reset_index(drop = True)
sample_df.head()
```

```python
# Select 5 secret values (canaries), test if they are not in the train dataset before insertion.
secrets = ["5601","1003","3456","7290","1342"]
sample_df.loc[sample_df["cc_num"].isin(secrets), "cc_num"]

```

```python
import numpy as np
from numpy.random import choice

weights = np.array([.05, .10, .15, .20, .50])

def create_canaries(df: pd.DataFrame, secrets, weights, frac=0.01) -> pd.DataFrame:
    """Insert secrets randomly into the location columns.
       These values should never be repeated by the model
    """
    weights /= weights.sum()
    cols = ['cc_num']
    # Remove the random state in the blueprint
    canaries = df.sample(frac=frac)
    for i, row in canaries.iterrows():
         canaries.at[i, choice(cols)] = choice(secrets, p=weights)
    return canaries
        
 
canaries = create_canaries(sample_df, secrets, weights, 0.01)
canaries.head()

```

```python
# Get the counts for each secret value
canaries["cc_num"].value_counts()
```

```python
from sklearn.utils import shuffle

# canary_sample_df = sample_df.append(canaries)
train_df = shuffle(sample_df.append(canaries),random_state=42).reset_index(drop =True)
# The last four digits of a credit card number might start with 0 and be removed when saved as integer. We save it as a string by inserting a single letter at first part of it.
train_df["cc_num"] = train_df["cc_num"].apply(lambda x:"m"+x)
train_df.to_csv("train.csv", index=False)
```

```python
from gretel_client.projects.models import read_model_config
import json

# Create model configuration for the DP model.
config = read_model_config("synthetics/default")


config['models'][0]["synthetics"]["params"]["vocab_size"] = 0
config['models'][0]["synthetics"]["params"]["epochs"] = 50
config['models'][0]["synthetics"]["params"]["learning_rate"] = 0.001
config['models'][0]["synthetics"]["params"]["batch_size"] = 4
config['models'][0]["synthetics"]["params"]["predict_batch_size"] = 1

# Enable Differential Privacy:
config['models'][0]["synthetics"]["params"]["dp"] = True
config['models'][0]["synthetics"]["params"]["dp_noise_multiplier"] = 0.001
config['models'][0]["synthetics"]["params"]["dp_l2_norm_clip"] = 2   # set low to demonstrate gradient clipping


#Setting the privacy filters off, since we are already using DP.
config["models"][0]['synthetics']['privacy_filters']["outliers"] = None
config["models"][0]['synthetics']['privacy_filters']["similarity"] = None

# DP configuration setting summary:
config_dict = config["models"][0]["synthetics"]["params"]
pd.DataFrame.from_dict(config_dict,orient="index",columns=["values"])


```

```python
# Create a project
from gretel_client.helpers import poll
from gretel_client.projects import create_or_get_unique_project

project = create_or_get_unique_project(name="cc-dp-model")
model = project.create_model_obj(model_config=config, data_source="train.csv")
model.submit_cloud()
poll(model)
```

```python
# Read the generated synthetis data from the synthetic model:
synthetic_df = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic_df.head()
```

```python
# Find the canaries in the synthetic data that were memorized by the model
string_secrets = ["m"+s for s in secrets]


def find_canaries(df, secrets):
    frequency = []
    raw = df.to_string()
    for secret in secrets:
      frequency.append(raw.count(str(secret)))
    return frequency

results = pd.DataFrame({"Secret value": string_secrets,
                        "Insertion count": find_canaries(train_df, string_secrets),
                        "Repetition by synthetic model" :find_canaries(synthetic_df, string_secrets)})

results
```
