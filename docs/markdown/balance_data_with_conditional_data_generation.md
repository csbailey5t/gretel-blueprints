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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/balance_data_with_conditional_data_generation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="UTRxpSlaczHY" -->
# Balancing datasets with conditional data generation

Imbalanced datasets are a common problem in machine learning. There are several different scenarios where an imbalanced dataset can lead to a less than optimal model solution. One scenario is when you're training a multi-class classifier and one or more of the classes have fewer training examples than the others. This can sometimes lead to a model that may look like it's doing well overall,when really the accuracy of the underepresented classes is inferior to that of the classes with good representation.

Another scenario is when the training data has imbalanced demographic data. Part of what the Fair AI movement is about is ensuring that AI models do equally well on all demographic slices.

One approach to improve representational biases in data is through by conditioning Gretel's synthetic data model to generate more examples of different classes of data.

You can use the approach to replace the original data with a balanced synthetic dataset or you can use it to augment the existing dataset, producing just enough synthetic data such that when added back into the original data, the imbalance is resolved.

In this notebook, we're going to step you through how to use Gretel synthetics to resolve demographic bias in a dataset. We will be creating a new synthetic dataset that can be used in place of the original one.

<!-- #endregion -->

<!-- #region id="An3JaXtu_15j" -->
## Begin by authenticating

<!-- #endregion -->

```python id="VEM6kjRsczHd"
%%capture
!pip install -U gretel-client
```

```python colab={"base_uri": "https://localhost:8080/"} id="ZQ-TmAdwczHd" outputId="4a8c2b52-950a-4c07-d9ee-b80293238f43"
# Specify your Gretel API key

import pandas as pd
from gretel_client import configure_session

pd.set_option("max_colwidth", None)

configure_session(api_key="prompt", cache="yes", validate=True)

```

<!-- #region id="dDfOuvA5_15n" -->
## Load and view the dataset

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="YRTunFZ2_15n" outputId="dc403944-03f8-4007-f47a-1d38eb1e81e9"
a = pd.read_csv(
    "https://gretel-public-website.s3.amazonaws.com/datasets/experiments/healthcare_dataset_a.csv"
)

a

```

<!-- #region id="sLkVPQlh_15o" -->
## Isolate the fields that require balancing

- We'll balance "RACE", "ETHNICITY", and "GENDER"

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="XN-KytoT_15p" outputId="8d40c38d-80b7-4613-c206-e3d889c8cf69"
a["RACE"].value_counts()

```

```python colab={"base_uri": "https://localhost:8080/"} id="sqpSM_EU_15q" outputId="aba9a196-68ec-403d-b47f-9f4a358dc669"
a["ETHNICITY"].value_counts()

```

```python colab={"base_uri": "https://localhost:8080/"} id="xZZ7v8Hf_15q" outputId="3358425a-5d46-43a4-ad51-0f7915f463cb"
a["GENDER"].value_counts()

```

<!-- #region id="1Eisd9JU_15r" -->
## Create a seed file

- Create a csv with one column for each balance field and one record for each combination of the balance field values.
- Replicate the seeds to reach the desired synthetic data size.

<!-- #endregion -->

```python id="iOi2i3qr_15s"
import itertools

# Choose your balance columns
balance_columns = ["GENDER", "ETHNICITY", "RACE"]

# How many total synthetic records do you want
gen_lines = len(a)

# Get the list of values for each seed field and the
# overall percent we'll need for each seed value combination
categ_val_lists = []
seed_percent = 1
for field in balance_columns:
    values = set(pd.Series(a[field].dropna()))
    category_cnt = len(values)
    categ_val_lists.append(list(values))
    seed_percent = seed_percent * 1 / category_cnt
seed_gen_cnt = seed_percent * gen_lines

# Get the combo seeds we'll need. This is all combinations of all
# seed field values
seed_fields = []
for combo in itertools.product(*categ_val_lists):
    seed_dict = {}
    i = 0
    for field in balance_columns:
        seed_dict[field] = combo[i]
        i += 1
    seed = {}
    seed["seed"] = seed_dict
    seed["cnt"] = seed_gen_cnt
    seed_fields.append(seed)

# Create a dataframe with the seed values used to condition the synthetic model
gender_all = []
ethnicity_all = []
race_all = []
for seed in seed_fields:
    gender = seed["seed"]["GENDER"]
    ethnicity = seed["seed"]["ETHNICITY"]
    race = seed["seed"]["RACE"]
    cnt = seed["cnt"]
    for i in range(int(cnt)):
        gender_all.append(gender)
        ethnicity_all.append(ethnicity)
        race_all.append(race)

df_seed = pd.DataFrame(
    {"GENDER": gender_all, "ETHNICITY": ethnicity_all, "RACE": race_all}
)

# Save the seed dataframe to a file
seedfile = "/tmp/balance_seeds.csv"
df_seed.to_csv(seedfile, index=False, header=True)

```

<!-- #region id="VVaGfSFc_15t" -->
## Create a synthetic config file

<!-- #endregion -->

```python id="BInkOazF_15u"
# Grab the default Synthetic Config file
from gretel_client.projects.models import read_model_config

config = read_model_config("synthetics/default")

```

```python id="Z3hDdxFn_15u"
# Adjust the desired number of synthetic records to generated

config["models"][0]["synthetics"]["generate"]["num_records"] = len(a)

```

```python id="uneHBVfN_15v"
# Adjust params for complex dataset

config["models"][0]["synthetics"]["params"]["data_upsample_limit"] = 10000

```

<!-- #region id="RR0AHEBR_15v" -->
## Include a seeding task in the config

<!-- #endregion -->

```python id="Qq-wkWq0_15v"
task = {"type": "seed", "attrs": {"fields": balance_columns}}
config["models"][0]["synthetics"]["task"] = task

```

<!-- #region id="IbDnimMH_15w" -->
## Train a synthetic model

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Yvf9EI85_15w" outputId="bcbed207-3a60-478a-9e65-88d54a45c9b2"
from gretel_client import projects
from gretel_client.helpers import poll

training_path = "training_data.csv"
a.to_csv(training_path)

project = projects.create_or_get_unique_project(name="balancing-data-example")
model = project.create_model_obj(model_config=config, data_source=training_path)

model.submit_cloud()
poll(model)

```

<!-- #region id="X--V8DHl_15w" -->
## Generate data using the balance seeds

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="PeZPWdXT_15x" outputId="ec54477f-a64d-4686-f7ce-9a4b355ed53f"
rh = model.create_record_handler_obj(
    data_source=seedfile, params={"num_records": len(df_seed)}
)
rh.submit_cloud()
poll(rh)
synth_df = pd.read_csv(rh.get_artifact_link("data"), compression="gzip")
synth_df.head()

```

<!-- #region id="GFoJ8niJ_15x" -->
## Validate the balanced demographic data

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CXdorzf1_15x" outputId="6732a6b0-b72f-48e0-db74-b7b0cdc40ff4"
synth_df["GENDER"].value_counts()

```

```python colab={"base_uri": "https://localhost:8080/"} id="yxrQujl0_15x" outputId="69ef1869-865e-4cff-e51e-c3447778619c"
synth_df["ETHNICITY"].value_counts()

```

```python colab={"base_uri": "https://localhost:8080/"} id="Ghc2mEQg_15y" outputId="710efabf-b480-4dbb-f145-2b717c6a5a11"
synth_df["RACE"].value_counts()

```

```python id="5152iEX1_15y"

```
