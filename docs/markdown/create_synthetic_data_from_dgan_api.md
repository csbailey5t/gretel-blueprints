---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3.9.10 64-bit ('3.9.10')
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/create_synthetic_data_from_dgan_api.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

<!-- #region id="NYS8U5A1KSFq" -->
**Creating Synthetic Time Series Data with DoppelGANger**


This Blueprint demonstrates how to create synthetic time series data via Gretel API with DoppelGANger (DGAN). The notebook provides a step-by-step process on how to take a raw dataframe and generate high-quality synthetic time series data. Specifically, we take a dataset containing daily prices over the past 35 years of two different oils (WTI and Brent) and show how to:


1.   Load and manipulate the dataset so that it is in the correct format for DGAN
2.   Set up a training configuration file for the Gretel API 
3.   Submit the model for training and monitor status
4.   Visuale and compare the synthetic and real data

<!-- #endregion -->

```python id="a513acf2"
# Install the required packages

%%capture
!pip install gretel_client pandas matplotlib numpy scipy torch
```

```python id="e85467d2"
import math

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yaml

from gretel_client import configure_session
from gretel_client.helpers import poll
from gretel_client.projects.projects import get_project
from gretel_client.projects.models import read_model_config

from plotly.subplots import make_subplots
```

```python id="48d0e9f3"
# Specify your Gretel API Key
configure_session(api_key="prompt", cache="no", validate=True)
```

```python id="900e0942"
# Download and load the oil data that we will generate synthetic data for

def get_oil():
    wti = pd.read_csv("https://datahub.io/core/oil-prices/r/wti-daily.csv")
    brent = pd.read_csv("https://datahub.io/core/oil-prices/r/brent-daily.csv")
    wti.columns = ["Date", "WTI Price"]
    brent.columns = ["Date", "Brent Price"]
    oil = wti.merge(brent)
    return oil
df = get_oil()
df
```

```python
# Plot entire 35 years of price history

COLUMNS = ["WTI Price", "Brent Price"]
TIME_COLUMN = "Date"
MAX_SEQUENCE_LEN = 10


for c in COLUMNS:
    plt.plot(pd.to_datetime(df[TIME_COLUMN]), df[c])
    plt.xlabel("Date")
    plt.ylabel(c)
    plt.xticks(rotation=90)
    plt.show()
```

```python id="Mq5UOoAokH1W"
# Plot several 10-day sequences from real data
# These correspond to training examples for the DGAN model
def plot_subsequence(df, max_sequence_len, index):
    local_df = df.iloc[index * max_sequence_len:(index + 1) * max_sequence_len, :]

    for c in COLUMNS:
        plt.plot(local_df[TIME_COLUMN], local_df[c], label=c)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=90)
    plt.show()

plot_subsequence(df, MAX_SEQUENCE_LEN, 20)
plot_subsequence(df, MAX_SEQUENCE_LEN, 500)
plot_subsequence(df, MAX_SEQUENCE_LEN, 731)
```

```python id="112988bf"
# Setup config and train model

TMP_FILE = "tmp_train.csv"

project = get_project(display_name="DGAN-oil", create=True)

print(f"Follow model training at: {project.get_console_url()}")

config = read_model_config("synthetics/time-series")
config["name"] = "dgan-oil-data"
config["models"][0]["timeseries_dgan"]["generate"] = {"num_records": 10000}


model = project.create_model_obj(model_config=config)

df.to_csv(TMP_FILE, index=False)
model.data_source = TMP_FILE

model.submit(upload_data_source=True)

poll(model)
```

```python id="v9CVE9S7Sqam"
# Read 10k synthetic examples
synthetic_df = pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
synthetic_df
```

```python
# Show first 20 rows of synthetic CSV
synthetic_df[0:20]
```

```python
# Helper functions for plotting

_GRETEL_PALETTE = ["#A051FA", "#18E7AA"]
_GRAPH_OPACITY = 0.75
_GRAPH_BARGAP = 0.2  # gap between bars of adjacent location coordinates
_GRAPH_BARGROUPGAP = 0.1  # gap between bars of the same location coordinates


def combine_subplots(
    figures: List[go.Figure],
    titles: List[str] = None,
    subplot_type: str = "xy",
    shared_xaxes=True,
    shared_yaxes=True,
) -> go.Figure:
    """
    Take a list of go.Figures and make a single go.Figure out of them.  They will all be on one row.
    Args:
        figures: List of go.Figures to combine.
        titles: List of subplot titles, must be same length as number of traces.
        subplot_type: see https://plotly.com/python/subplots/#subplots-types,
        shared_xaxes: Passed into plotly make_subplots call, see
            https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html
        shared_yaxes: Passed into plotly make_subplots call, see
            https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html
    Returns:
        a single new plotly.graph_objects.Figure.
    """
    specs = [[{"type": subplot_type}] * len(figures)]

    fig = make_subplots(
        rows=1,
        cols=len(figures),
        specs=specs,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        subplot_titles=titles,
    )
    for i, f in enumerate(figures):
        for t in f.select_traces():
            fig.add_trace(trace=t, row=1, col=i + 1)
        fig.layout.update(f.layout)
    return fig

def correlation_heatmap(matrix: pd.DataFrame, name: str = "Correlation") -> go.Figure:
    """
    Generate the figure for a list of correlation matrices.
    Arguments:
        matrix: The correlation matrix computed by dython.
        name: Name to use in add_trace.
    Returns:
        A plotly.graph_objects.Figure, a subplot with heatmaps.
    """
    fig = go.Figure()
    fields = [x if len(x) <= 15 else x[0:14] + "..." for x in matrix.columns]
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            y=fields,
            x=fields,
            xgap=1,
            ygap=1,
            coloraxis="coloraxis",
            name=name,
        )
    )
    fig.update_layout(
        coloraxis=dict(
            colorscale=[
                [0.0, "#E8F3C6"],
                [0.25, "#94E2BA"],
                [0.5, "#31B8C0"],
                [0.75, "#4F78B3"],
                [1.0, "#76137F"],
            ],
            cmax=1.0,
            cmin=0,
        ),
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    fig.update_yaxes(dtick=1)
    return fig

def histogram(left: pd.Series, right: pd.Series) -> Optional[go.Figure]:
    """
    Generate a histogram distplot for a numeric distribution.
    Arguments:
        left: The left pd.Series for which we make the histogram.
        right: The right pd.Series for which we make the histogram.
    Returns:
        A plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.update_layout(
        yaxis_title_text="Percentage",
        bargap=_GRAPH_BARGAP,
        bargroupgap=_GRAPH_BARGROUPGAP,
        showlegend=False,
    )

    left_copy = pd.Series(left)
    left_copy.dropna(inplace=True)
    right_copy = pd.Series(right)
    right_copy.dropna(inplace=True)

    if len(left_copy) == 0 or len(right_copy) == 0:
        return fig

    q1 = np.quantile(left_copy, 0.25)
    q3 = np.quantile(left_copy, 0.75)
    iqr = q3 - q1
    max_range = min(max(left_copy), (q3 + (1.5 * iqr)))
    min_range = max(min(left_copy), (q1 - (1.5 * iqr)))

    filtered_left_copy = [i for i in left_copy if min_range <= i <= max_range]
    filtered_right_copy = [i for i in right_copy if min_range <= i <= max_range]
    binsize = (max_range - min_range) / 30

    fig.add_trace(
        go.Histogram(
            x=filtered_left_copy,
            histnorm="percent",
            name="Training",
            xbins=dict(start=min_range, end=max_range, size=binsize),
            marker=dict(color=_GRETEL_PALETTE[0]),
            opacity=_GRAPH_OPACITY,
            hovertemplate="(%{x}, %{y:.2f})",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=filtered_right_copy,
            histnorm="percent",
            name="Synthetic",
            xbins=dict(start=min_range, end=max_range, size=binsize),
            marker=dict(color=_GRETEL_PALETTE[1]),
            opacity=_GRAPH_OPACITY,
            hovertemplate="(%{x}, %{y:.2f})",
        )
    )
    return fig

```

```python id="1bfac4a5"
# Compare correlations between variables in the real and synthetic data
print("Difference in real correlations and synethic data correlations:")
correlation_heatmap(df[COLUMNS].corr() - synthetic_df[COLUMNS].corr())
```

```python id="0FmWGdwkgmlD"
# Plot histograms of the distribution of values within each column
h1 = histogram(df['WTI Price'], synthetic_df['WTI Price'])
h2 = histogram(df['Brent Price'], synthetic_df['Brent Price'])
combine_subplots(
    figures=[h1, h2],
    titles=['WTI Price', 'Brent Price'],
    subplot_type = "xy",
    shared_xaxes=True,
    shared_yaxes=True,
)

```

```python id="095ed91d"
# Functions to calculate autocorrelation
def autocorr(X, Y):
    EPS = 1e-8
    Xm = torch.mean(X, 1).unsqueeze(1)
    Ym = torch.mean(Y, 1).unsqueeze(1)
    r_num = torch.sum((X - Xm) * (Y - Ym), 1)
    r_den = torch.sqrt(torch.sum((X - Xm)**2, 1) * torch.sum((Y - Ym)**2, 1))

    r_num[r_num == 0] = EPS
    r_den[r_den == 0] = EPS

    r = r_num / r_den
    r[r > 1] = 0
    r[r < -1] = 0

    return r
    
def get_autocorr(feature):
    feature = torch.from_numpy(feature)
    feature_length = feature.shape[1]
    autocorr_vec = torch.Tensor(feature_length-2)

    for j in range(1, feature_length - 1):
      autocorr_vec[j - 1] = torch.mean(autocorr(feature[:, :-j], feature[:, j:]))

    return autocorr_vec.cpu().detach().numpy()

def generate_numpy_for_autocorr(df, batch_size):
    features = df[COLUMNS].to_numpy()
    n = features.shape[0] // batch_size

    # Shape is now (# examples, # time points, # features)
    features = features[:(n*batch_size),:].reshape(-1, batch_size, features.shape[1])
    return features
```

```python id="9bb282d0"
# Generate autocorrelations from synthetic and real data and plot

acf = get_autocorr(generate_numpy_for_autocorr(df, MAX_SEQUENCE_LEN))
synthetic_acf = get_autocorr(generate_numpy_for_autocorr(synthetic_df, MAX_SEQUENCE_LEN))
# Figure 1, autocorrelation
plt.plot(acf, label="real", color=_GRETEL_PALETTE[0])
plt.plot(synthetic_acf, label="generated", color=_GRETEL_PALETTE[1])
plt.xlabel("Time lag (days)")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation of Oil Prices")
plt.legend()
plt.show()
```

```python id="rtr7B7vx72lN"
# Plot several 10-day sequences from synthetic data

plot_subsequence(synthetic_df, MAX_SEQUENCE_LEN, 5)
plot_subsequence(synthetic_df, MAX_SEQUENCE_LEN, 5000)
plot_subsequence(synthetic_df, MAX_SEQUENCE_LEN, 9121)

```
