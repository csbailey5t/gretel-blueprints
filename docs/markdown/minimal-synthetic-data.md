---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: 'Python 3.9.10 (''gretel'': venv)'
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/minimal-synthetic-data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

```python id="iovURYt3d_pa"
pip install -U gretel-client pandas
```

```python id="PryXC9MZd_pb"
# Specify your Gretel API key
import pandas as pd
from gretel_client import configure_session

configure_session(api_key="prompt", cache="yes", validate=True)
```

```python id="94NFYFbEd_pc"
# Create a project and set model configuration
from gretel_client.projects import create_or_get_unique_project
project = create_or_get_unique_project(name="mlworld")

from gretel_client.projects.models import read_model_config
config = read_model_config("synthetics/default")
```

```python id="gK2B5viId_pc"
# Load and preview the DataFrame to train the synthetic model on.
import pandas as pd

dataset_path = "https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/USAdultIncome5k.csv"
df = pd.read_csv(dataset_path)
df.to_csv("training_data.csv", index=False)
df
```

```python id="z1Ff1N3xd_pc"
from gretel_client.helpers import poll

model = project.create_model_obj(model_config=config, data_source="training_data.csv")
model.submit_cloud()

poll(model)
```

```python id="lDNP0xAid_pd"
# View the synthetic data

synthetic_df = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")

synthetic_df
```
