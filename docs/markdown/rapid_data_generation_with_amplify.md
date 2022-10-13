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
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/rapid_data_generation_with_amplify.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="sugXH-2KDYdE" -->
# Generate high volumes of data rapidly with Gretel Amplify

*   This notebook demonstrates how to **generate lots of data fast** using Gretel Amplify
*   To run this notebook, you will need an API key from the [Gretel console](https://console.gretel.cloud/dashboard).


<!-- #endregion -->

<!-- #region id="yOYfJXYREOSI" -->
## Getting Started

<!-- #endregion -->

```python id="VEM6kjRsczHd"
%%capture
!pip install -U gretel-client
```

```python id="kQYlGEMbDEBv"
# Imports
import json
import pandas as pd
from re import findall

from gretel_client import configure_session
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config
from gretel_client.helpers import poll
```

```python cellView="form" id="HWg6t3ko-I2-"
# @title
from re import findall


def get_output_stats(logs):
    i = len(logs)-1
    output_recs = 0
    while True:
        ctx = len(logs[i]['ctx'])
        if ctx != 0:
            output_recs = int(findall('\d*\.?\d+', logs[-4]['msg'])[0])
            output_size = logs[i]['ctx']['final_size_mb']
            gen_time = logs[i]['ctx']['amplify_time_min']*60
            throughput_MBps = logs[i]['ctx']['throughput_mbps']

            return(output_recs, output_size, gen_time, throughput_MBps)
            break
        i -= 1


def stats(model):

    # Statistics

    stats = get_output_stats(model.logs)

    target_size = model.model_config['models'][0]['amplify']['params']['target_size_mb']
    output_recs = stats[0]
    output_size = stats[1]
    time = model.billing_details['total_time_seconds']
    recs_per_sec = output_recs/time
    total_MBps = output_size/time
    gen_time = stats[2]
    gen_recs_per_sec = output_recs/gen_time
    throughput_MBps = stats[3]

    print('\033[1m' + "Statistics" '\033[0m')
    print("Target Size: \t\t{} MB".format(target_size))
    print("Output Rows: \t\t{} records".format(output_recs))
    print("Output Size: \t\t{:.2f} MB".format(output_size))
    print("Total Time: \t\t{:.2f} seconds".format(time))
    print("Total Speed: \t\t{:.2f} records/s".format(recs_per_sec))
    print("Total Speed: \t\t{:.2f} MBps".format(total_MBps))
    print("Generation Time: \t{:.2f} seconds".format(gen_time))
    print("Generation Speed: \t{:.2f} records/s".format(gen_recs_per_sec))
    print("Generation Speed: \t{:.2f} MBps".format(throughput_MBps))

```

```python id="rjBbbGyNO2PO"

pd.set_option("max_colwidth", None)

# Specify your Gretel API Key
configure_session(api_key="prompt", cache="no", validate=True)
```

<!-- #region id="2mXcFk2Cy0lC" -->
## Load and preview data

For this demo, we'll use a [US Census dataset](https://github.com/gretelai/gretel-blueprints/blob/main/sample_data/us-adult-income.csv) as our input data. This dataset contains 14,000 records, 15 fields, and is about 1.68 MB in size. 

If you want to use another dataset, just replace the URL. 
<!-- #endregion -->

```python id="Rgx85TgkPJsY"
url = 'https://raw.githubusercontent.com/gretelai/gretel-blueprints/main/sample_data/us-adult-income.csv'
df = pd.read_csv(url)
print('\033[1m'+ "Input Data - US Adult Income" +'\033[0m')
print('Number of records: {}'.format(len(df)))
print('Size: {:.2f} MB'.format(df.memory_usage(index=True).sum()/1e6))
df
```

<!-- #region id="2kKGDsEezMVY" -->
## Set target output size

There are two ways to indicate the amount of data your want to generate with Amplify. You can use the `num_records` config parameter to specify the number of records to produce. Or, you can use the `target_size_mb` parameter to designate the desired output size in megabytes. The maximum value for `target_size_mb` is 5000 (5GB). Only one parameter can be specified. To read more about the Amplify config, you can check out our docs [here](https://docs.gretel.ai/gretel.ai/synthetics/models/amplify).

In this example, we want to generate 5GB of data so we'll set the `target_size_mb` parameter to be `5000`.
<!-- #endregion -->

```python id="cpfJzWa8pENd"
# Pull Amplify model config 
config = read_model_config("https://raw.githubusercontent.com/gretelai/gretel-blueprints/main/config_templates/gretel/synthetics/amplify.yml")

# Set config parameters

config['models'][0]['amplify']['params']['target_size_mb'] = 5000        # 5 GB
config['name'] = "amplify-demo"
```

<!-- #region id="X19N2FOTxpEv" -->
## Create and run model
<!-- #endregion -->

```python id="GOIbGmCXtGS5"
# Designate project
project = create_or_get_unique_project(name="amplify")

# Create and submit model 
model = project.create_model_obj(model_config=config, data_source=df)
model.submit_cloud()
poll(model)
```

<!-- #region id="XdRDFW1izjuR" -->
## View results
<!-- #endregion -->

```python id="govCEdQ2VxU-"
stats(model)
```

```python id="CWUvcfzXvptx"
amp = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
amp
```
