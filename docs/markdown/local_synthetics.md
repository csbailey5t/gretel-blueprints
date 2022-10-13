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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/local_synthetics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

## Train a Gretel.ai synthetic data model locally

This notebook walks through training a model and generating synthetic data locally in your environment.

Follow the instructions here to set up your local environment and GPU: https://docs.gretel.ai/environment-setup

Prerequisites:

- Python 3.9+ (`python --version`).
- GPU with CUDA configured highly recommended (`nvidia-smi`).
- Ensure that Docker is running (`docker info`.
- The Gretel client SDK is installed and configured (`pip install -U gretel-client; gretel configure`).


```python
import json

from smart_open import open
import pandas as pd

from gretel_client import submit_docker_local
from gretel_client.projects import create_or_get_unique_project

data_source = "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"

```

```python
# Load and preview the DataFrame to train the synthetic model on.

df = pd.read_csv(data_source)
df.to_csv("training_data.csv", index=False)
df

```

```python
# Load config and set training parameters
from gretel_client.projects.models import read_model_config

config = read_model_config("synthetics/default")

config["models"][0]["synthetics"]["params"]["epochs"] = 50
config["models"][0]["synthetics"]["data_source"] = "training_data.csv"

print(json.dumps(config, indent=2))

```

```python
# Create a project and train the synthetic data model

project = create_or_get_unique_project(name="synthetic-data-local")
model = project.create_model_obj(model_config=config)
run = submit_docker_local(model, output_dir="tmp/")

```

```python
# View the generated synthetic data

synthetic_df = pd.read_csv("tmp/data_preview.gz", compression="gzip")
synthetic_df

```

```python
# View report that shows the statistical performance between the training and synthetic data

import IPython

IPython.display.HTML(data=open("tmp/report.html.gz").read(), metadata=dict(isolated=True))

```

```python
# Use the trained model to create additional synthetic data

record_handler = model.create_record_handler_obj(params={"num_records": 100})

run = submit_docker_local(
    record_handler, model_path="tmp/model.tar.gz", output_dir="tmp/"
)

```

```python
synthetic_df_new = pd.read_csv("tmp/data.gz", compression="gzip")
synthetic_df_new

```
