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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/classify_freetext_with_nlp.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="aIBl7LPg0Zzc" -->
# Using Gretel Classify to Label Free Text

In this blueprint, we analyze and label a set of Yelp reviews looking for PII and other potentially sensitive information.

<!-- #endregion -->

<!-- #region id="5zlWDeUZ0Zzd" -->
## Setup

First we install our python dependencies and configure the Gretel client.

_Note: we install spacy for their visualization helper, displacy_

<!-- #endregion -->

```python id="mmcTAKie0Zze"
!pip install -Uqq gretel-client spacy datasets
```

```python id="6DuZ3OP-0Zzf"
import json
import datasets
import pandas as pd
from gretel_client import poll, configure_session
from gretel_client.projects import create_or_get_unique_project

pd.set_option("max_colwidth", None)

dataset_file_path = "reviews.csv"

configure_session(api_key="prompt", cache="yes", validate=True)

```

<!-- #region id="kDNRpc-l0Zzf" -->
## Load the dataset

Using Hugging Face's [datasets](https://github.com/huggingface/datasets) library, we load a dataset containing a dump of [Yelp reviews](https://huggingface.co/datasets/yelp_review_full). This data contains unstructured review text that we pass through a NER pipeline for labeling and PII discovery.

<!-- #endregion -->

```python id="dw1QMDr40Zzg"
source_dataset = datasets.load_dataset("yelp_review_full")
source_df = pd.DataFrame(source_dataset["train"]).sample(n=300, random_state=99)
source_df.to_csv(dataset_file_path, index=False)

```

```python id="2ywqtTxf0Zzh"
source_df.head()

```

<!-- #region id="HoClMb1E0Zzh" -->
## Configure a Gretel Project and Model

<!-- #endregion -->

```python id="7VI-uVY70Zzi"
project = create_or_get_unique_project(name="Gretel NLP Yelp Reviews")

```

```python id="S0l3tqQX0Zzi"
# Passing `use_nlp: true` into the model config,
# enables additional predictions using NLP models.
classify_config = """
schema_version: "1.0"
models:
  - classify:
      data_source: "_"
      use_nlp: true
"""

```

<!-- #region id="2pvl7qbr0Zzj" -->
If you wish to transform the dataset instead, you may pass the same `use_nlp: true` property into a transformation pipeline. For an example of a transform pipeline, see the [Redact PII Notebook](https://github.com/gretelai/gretel-blueprints/blob/main/docs/notebooks/redact_pii.ipynb). Below is an example that uses nlp.

```yaml
schema_version: "1.0"
models:
  - transforms:
      data_source: "_"
      use_nlp: true
      policies:
        - name: remove_pii
          rules:
            - name: redact_pii
              conditions:
                value_label:
                  - person_name
                  - location
                  - credit_card_number
                  - phone_number
                  - email_address
              transforms:
                - type: fake
                - type: redact_with_char
                  attrs:
                    char: X
```

<!-- #endregion -->

<!-- #region id="R1LAysyo0Zzj" -->
### Create the Classification Model

This next cell will create the classification model. After we verify the model is working correctly, the the entire dataset will be passed into the model for classification.

<!-- #endregion -->

```python id="a0jjxsWu0Zzk"
model = project.create_model_obj(yaml.safe_load(classify_config), dataset_file_path)
model.submit_cloud()
poll(model)

```

<!-- #region id="By5EcgYP0Zzk" -->
Using the created model, we download the report to get a summary view of found entities. This report is based on a sample of the original dataset, and is used to ensure the model has been configured correctly.

<!-- #endregion -->

```python id="D76WDvpM0Zzk"
# `report_json` contains a summary of entities by field
with open(model.get_artifact_link("report_json")) as fh:
    report = json.load(fh)

```

```python id="25FPWSd40Zzl"
# By converting these summaries into a dataframe we can quickly view
# entities found by the model.
summary = []
for field in report["metadata"]["fields"]:
    row = {"name": field["name"]}
    for entity in field["entities"]:
        row[entity["label"]] = entity["count"]
    summary.append(row)

pd.DataFrame(summary).set_index("name").fillna(0)

```

<!-- #region id="otkUQlrf0Zzl" -->
### Classify the reviews

Now that the model has been configured and verified, let's run the full dataset through the model.

<!-- #endregion -->

```python id="vQz1bRNc0Zzl"
records = model.create_record_handler_obj(data_source=dataset_file_path)
records.submit_cloud()
poll(records)

```

```python id="8dJdAA2C0Zzm"
# the `data` artifact returns a JSONL formatted file containing
# entity predictions by row.
with open(records.get_artifact_link("data")) as fh:
    records = [json.loads(line) for line in fh]

```

```python colab={"background_save": true} id="AN2mQqyW0Zzm"
from IPython.display import clear_output
from spacy import displacy


for row, entities in zip(source_df.values, records):
    label, text = row

    colors = {}
    palette = [
        "#7aecec",
        "#bfeeb7",
        "#feca74",
        "#ff9561",
        "#aa9cfc",
        "#c887fb",
        "#9cc9cc",
        "#ffeb80",
        "#ff8197",
        "#ff8197",
        "#f0d0ff",
        "#bfe1d9",
        "#e4e7d2",
    ]

    for index, label in enumerate([x["label"] for x in entities["entities"]]):
        colors[label.upper()] = palette[index % len(palette)]

    options = {"ents": list(colors.keys()), "colors": colors}

    displacy.render(
        {
            "text": text,
            "ents": [e for e in entities["entities"] if e["field"] == "text"],
        },
        style="ent",
        jupyter=True,
        manual=True,
        options=options,
    )
    input("\nPress [enter] to see the next review")

```

```python id="eIraRtkI0Zzm"
# now that you've run the notebook, you can also view the same
# project using our web console.
project.get_console_url()

```
