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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/synthetic_data_uber_differential_privacy.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="UTRxpSlaczHY" -->
# A differentially private, synthetic ride-share dataset

This blueprint utilizes Gretel's SDKs to create a synthetic version of your own data. Our SDKs create automatic data validators to help ensure the data generated has the same semantics as the source data. Additionally, the SDKs do autmoatic header clustering to help maintain statistical relations between columns.
<!-- #endregion -->

```python id="VEM6kjRsczHd"
%%capture
!pip install gretel-client 
```

```python id="ZQ-TmAdwczHd"
# Load your Gretel API key. You can acquire this from the Gretel Console 
# @ https://console.gretel.cloud

import pandas as pd
from gretel_client import configure_session

pd.set_option('max_colwidth', None)
configure_session(api_key="prompt", cache="yes", validate=True)
```

```python id="TzxieDJbgvW7"
# Read the training dataset before inserting canary values:
dataset_path = "https://gretel-public-website.s3.amazonaws.com/datasets/uber_scooter_rides_1day.csv"
df = pd.read_csv(dataset_path,names = ["hour","bike_id","src_lat","src_lon","dst_lat","dst_lon"]).round(5)
df.head()
```

```python id="nSIlKuSCk1kj"
from numpy.random import uniform
import numpy as np
from numpy.random import choice
 
# Create random secrets (canaries) to insert into training set
secrets = [85.31243, 80.71705, 84.98992, 63.20242]
weights = np.array([.05, .15, .30, .50])

def create_canaries(df: pd.DataFrame, secrets, weights, frac=0.01) -> pd.DataFrame:
    """Insert secrets randomly into the location columns.
       These values should never be repeated by the model
    """
    weights /= weights.sum()
    cols = ['src_lon', 'src_lat', 'dst_lon', 'dst_lat']
    
    canaries = df.sample(frac=frac, random_state=42)
    for i, row in canaries.iterrows():
         canaries.at[i, choice(cols)] = choice(secrets, p=weights)
    return canaries
        
 
canaries = create_canaries(df, secrets, weights, 0.01)
canaries.head()
```

```python id="3RyDm9V5MDtR"
train_df = df.append(canaries,ignore_index= True)
# shuffle the training dataset with appended canary values before training the model:
from sklearn.utils import shuffle
train_df = shuffle(train_df,random_state=42).reset_index(drop =True)
# Save the dataset in a csv to train the model with.
train_df.to_csv("train.csv", index=False)
train_df.head()

```

```python id="9hfXq5gMhByJ"
from gretel_client.projects.models import read_model_config

# Create model configuration.
config = read_model_config("synthetics/default")

config['models'][0]["synthetics"]["params"]["vocab_size"] = 0
config['models'][0]["synthetics"]["params"]["epochs"] = 50
config['models'][0]["synthetics"]["params"]["learning_rate"] = 0.001  # set low to demonstrate gradient clipping
config['models'][0]["synthetics"]["params"]["batch_size"] = 4
config['models'][0]["synthetics"]["params"]["predict_batch_size"] = 1

# Enable Differential Privacy:
config['models'][0]["synthetics"]["params"]["dp"] = True
config['models'][0]["synthetics"]["params"]["dp_noise_multiplier"] = 0.001
config['models'][0]["synthetics"]["params"]["dp_l2_norm_clip"] = 1.5

#Setting the privacy filters off, since we are already using DP.
config["models"][0]['synthetics']['privacy_filters']["outliers"] = None
config["models"][0]['synthetics']['privacy_filters']["similarity"] = None

seed_columns = ["hour", "bike_id"]
task = {"type": "seed", "attrs": {"fields": seed_columns}}
config["models"][0]["synthetics"]["task"] = task

# DP configurationsetting summary:
data = config["models"][0]["synthetics"]["params"]
pd.DataFrame.from_dict(data,orient="index",columns=["values"])
```

```python id="CCW-JaiNczHf"
# Create a project
from gretel_client.helpers import poll
from gretel_client.projects import create_or_get_unique_project

project = create_or_get_unique_project(name="ride-share-DP-Model")
model = project.create_model_obj(model_config=config, data_source="train.csv")
model.submit_cloud()
poll(model)
```

```python id="srW1HBA-d3Mp"
# Read the synthetic data created from the conditioned synthetic data model.
synthetic_df = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic_df.head()
```

```python id="W5BhlCaoKGhn"
# Find the canaries that were replayed by our model
def find_canaries(df, secrets):
    frequency = []
    raw = df.to_string()
    for secret in secrets:
      frequency.append(raw.count(str(secret)))
    return frequency

results = pd.DataFrame({"Secret value": secrets,
                        "Insertion count": find_canaries(train_df, secrets),
                        "Repetition by synthetic model" :find_canaries(synthetic_df, secrets)})

results

```
