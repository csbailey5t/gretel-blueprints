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
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/conditional_text_generation_with_gpt.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="UTRxpSlaczHY" -->
# Generating Synthetic Text

This notebook will walk you through generating realistic but synthetic text examples using an open-source implementation of OpenAI's GPT-3 architecture. 

In this example, we will generate new annotated text utterances that can be used to augment a real world financial dataset called `banking77`. This augmented dataset will have additional annotated examples that can help downstream ML models better understand and respond to new customer queries. To run this notebook, you will need an API key from the Gretel console,  at https://console.gretel.cloud. 
<br>

** **Limitations and Biases** **
Large-scale language models such as GPT-X may produce untrue and/or offensive content without warning. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results. For more information and examples please see [OpenAI](https://huggingface.co/gpt2#limitations-and-bias) and [EleutherAI](https://huggingface.co/EleutherAI/gpt-neo-125M#limitations-and-biases)'s docs for more details.
<!-- #endregion -->

```python id="VEM6kjRsczHd"
%%capture
!pip install -U gretel-client
```

<!-- #region id="rhBCe4PfrTWW" -->
## Set up your project
* `DATASET_PATH`: Specify a dataset to run on.
* `INTENT`: Select an intent from the training data to boost examples for.
* `SEPARATOR`: Specify a separator character (default=`,`) to combine intents and texts with into a single column.
* `PROJECT`: Specify a project name.
<!-- #endregion -->

```python id="ZQ-TmAdwczHd"
import json

import pandas as pd
from gretel_client import configure_session
from gretel_client.helpers import poll
from gretel_client.projects import create_or_get_unique_project, get_project


DATASET_PATH = 'https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/banking77.csv'
INTENT = "card arrival"
SEPARATOR = ','
PROJECT = 'banking77'
```

```python id="ZOygU-1MrTWY" colab={"base_uri": "https://localhost:8080/"} outputId="4fc3ff59-dd7a-49a9-9c36-2faec13f5d91"
# Log into Gretel and configure project

configure_session(api_key="prompt", cache="yes", endpoint="https://api.gretel.cloud", validate=True, clear=True)

project = create_or_get_unique_project(name=PROJECT)
project
```

<!-- #region id="4PD5B0U06ALs" -->
## Create the model configuration

In this notebook we will use GPT-Neo, a transformer model designed using EleutherAI's replication of OpenAI's GPT-3 Architecture. This model has been pre-trained on the Pile, a large-scale dataset using 300 billion tokens over 572,300 steps. In this introductory example, we will fine-tune GPT-Neo to generate synthetic text utterances for a given intent that could be used to train a chat-bot.
<!-- #endregion -->

```python id="3n0npt-_rTWa"
config = {
  "schema_version": 1,
  "models": [
    {
      "gpt_x": {
        "data_source": "__",
        "pretrained_model": "EleutherAI/gpt-neo-125M",
        "batch_size": 4,
        "epochs": 1,
        "weight_decay": 0.1,
        "warmup_steps": 100,
        "lr_scheduler": "cosine",
        "learning_rate": 1e-6
      }
    }
  ]
}
```

<!-- #region id="s9LTh7GO6VIu" -->
## Load and preview the training dataset
Create single-column CSV training set by combining `intent` + `SEPARATOR` + `text`.

<!-- #endregion -->

```python id="YMg9nX6SczHe" colab={"base_uri": "https://localhost:8080/", "height": 424} outputId="f318c986-7ee8-4c36-cccb-4b30e5f05deb"
def create_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Combine intents and text into a single string to pass to GPT-X.
    """
    records = []
    max_tokens = 0
    
    df = pd.read_csv(dataset_path)
    df['intent_and_text'] = df['intent'] + SEPARATOR + df['text']
    return df
    

pd.set_option('max_colwidth', None)

df = create_dataset(DATASET_PATH)
df[['intent_and_text']].to_csv('finetune.csv', index=False)
df
```

<!-- #region id="WxnH8th-65Dh" -->
## Train the synthetic model
In this step, we will task the worker running in the Gretel cloud, or locally, to fine-tune the GPT language model on the source dataset.
<!-- #endregion -->

```python id="O4-E_F0qczHe" colab={"base_uri": "https://localhost:8080/"} outputId="d10f75c7-f21d-42a5-b6a7-35164b5e2f6b"
%%time 

model = project.create_model_obj(model_config=config)
model.data_source = "finetune.csv"
model.name = f"{PROJECT}-gpt"
model.submit_cloud()

poll(model)
```

<!-- #region id="6IkWOnVQ7oo1" -->
## Generate synthetic text data
The next cells walk through sampling data from the fine-tuned model using a prompt (conditional data generation). 
<!-- #endregion -->

```python id="C8gD9Q2XSFyv" colab={"base_uri": "https://localhost:8080/"} outputId="5e83f53e-495f-408e-91c7-a1c26d623471"
# Generate new text examples for a given intent by seeding
# model generation with examples from the class. Hint: We have found
# prompting the model with ~25 examples for the class you wish to 
# generate to work well in practice.

def create_prompt(df: pd.DataFrame, intent: str = "", recs: int = 25) -> str:
    """
    Seed GPT-X text generation with an intent from the training data.
    """
    sample = df.query(f'intent == "{intent}"').head(recs)
    prompt = "\n".join([x[0] for x in sample[['intent_and_text']].values])
    return prompt


prompt = create_prompt(df=df, intent=INTENT, recs=25)

record_handler = model.create_record_handler_obj(
    params={"num_records": 1, 
            "maximum_text_length": 1000, 
            "prompt": prompt}
)
record_handler.submit_cloud()
poll(record_handler)
```

<!-- #region id="Xy8q3f2dTAHv" -->
# Creating synthetic intents

In the cell below, we process the raw texts generated by GPT-X into a structured dataframe format, by splitting each row based on the intent prefix (`card_arrival`) that was used to prompt generation.


<!-- #endregion -->

```python id="8Fx4aeMOSFyw" colab={"base_uri": "https://localhost:8080/", "height": 520} outputId="d8416537-56e2-400e-bda5-933685f4b3ac"
def get_intents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract new intents generated by the GPT-X model.
    """
    MIN_LENGTH = 20
    texts = []
    
    for idx, row in gptx_df.iterrows(): 
        for text in row[0].split(f"{INTENT}{SEPARATOR}"):
            text = text.strip()
            if len(text) > MIN_LENGTH:
                texts.append([INTENT, text])

    intents = pd.DataFrame(texts, columns=['intent', 'synthetic_text'])
    return intents


gptx_df = pd.read_csv(record_handler.get_artifact_link("data"), compression='gzip')
syn = get_intents(df=gptx_df)
syn.head(15)

```
