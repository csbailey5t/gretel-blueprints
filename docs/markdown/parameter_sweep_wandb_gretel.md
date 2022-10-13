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

<!-- #region id="nwFPp7oB-qe5" -->
<img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
<br>
<img src="https://www.gitbook.com/cdn-cgi/image/height=40,fit=contain,dpr=1,format=auto/https%3A%2F%2F2196202216-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252F-MW662bNvw1TgbuEBiwQ%252Flogo%252FLzl7Qs5X5sYkFBgjygeZ%252Fgretel_brand_wordmark_padded%25403x.png%3Falt%3Dmedia%26token%3D3f02fe4f-8684-443e-8aea-83a0e512cd96" width="200" alt="Gretel.ai" />

<!-- #endregion -->

<!-- #region id="TJ8re86X-qe-" -->
<a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!-- #endregion -->

```python id="hlD2ZWWPBsbZ"
%%capture
!pip install -U wandb gretel_client
```

```python colab={"base_uri": "https://localhost:8080/"} id="qWtbl6MC-qfA" outputId="2bd7eaf9-e4f9-4b77-be81-39eb64ecde6f"
import wandb

wandb.login()

```

```python colab={"base_uri": "https://localhost:8080/"} id="wxQkrQEYORkb" outputId="86fd83a7-b241-42f4-d8fb-0f64a8396180"
from gretel_client import configure_session

configure_session(api_key="prompt", cache="yes", validate=True)

```

<!-- #region id="iw8w2S3N-qfB" -->
### 👈 Pick a `method`

<!-- #endregion -->

<!-- #region id="BettZjD_-qfC" -->
The first thing we need to define is the `method`
for choosing new parameter values.

We provide the following search `methods`:

- **`grid` Search** – Iterate over every combination of hyperparameter values.
  Very effective, but can be computationally costly.
- **`random` Search** – Select each new combination at random according to provided `distribution`s. Surprisingly effective!
- **`bayes`ian Search** – Create a probabilistic model of metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric. Works well for small numbers of continuous parameters but scales poorly.

We'll stick with `random`.

<!-- #endregion -->

```python id="V8L1Atgk-qfD"
sweep_config = {"method": "bayes"}

```

<!-- #region id="Hwj0REZq-qfD" -->
For `bayes`ian Sweeps,
you also need to tell us a bit about your `metric`.
We need to know its `name`, so we can find it in the model outputs
and we need to know whether your `goal` is to `minimize` it
(e.g. if it's the squared error)
or to `maximize` it
(e.g. if it's the accuracy).

<!-- #endregion -->

```python id="YXmA2x7S-qfE"
metric = {"name": "sqs", "goal": "maximize"}

sweep_config["metric"] = metric

```

<!-- #region id="R_55ejha-qfE" -->
If you're not running a `bayes`ian Sweep, you don't have to,
but it's not a bad idea to include this in your `sweep_config` anyway,
in case you change your mind later.
It's also good reproducibility practice to keep note of things like this,
in case you, or someone else,
come back to your Sweep in 6 months or 6 years
and don't know whether `val_G_batch` is supposed to be high or low.

<!-- #endregion -->

<!-- #region id="vODALaVA-qfE" -->
### 📃 Name the hyper`parameters`

<!-- #endregion -->

```python id="4jWhLM5t-qfF"
parameters_dict = {
    "epochs": {"values": [25, 50, 100, 150]},
    "learning_rate": {"values": [0.001, 0.005, 0.01]},
    "vocab_size": {"values": [0, 500, 1000, 10000, 20000]},
    "rnn_units": {"values": [64, 256, 1024]},
    "batch_size": {"values": [64, 256]},
}

sweep_config["parameters"] = parameters_dict

```

```python colab={"base_uri": "https://localhost:8080/"} id="c3POc4m--qfH" outputId="2a3d2a5a-6352-4ad4-94a0-ad31162c972c"
import pprint

pprint.pprint(sweep_config)

```

<!-- #region id="0LhYjh6n-qfH" -->
# Step 2️⃣. Initialize the Sweep

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 424} id="VAZN0apOFb5D" outputId="e9a07cfa-0c2f-4ce3-bd2c-a397d7009cb5"
import pandas as pd

# Load the training dataset
dataset_path = "https://gretel-public-website.s3.amazonaws.com/datasets/credit-timeseries-dataset.csv"
df = pd.read_csv(dataset_path)
df.to_csv("training_data.csv", index=False)
df

```

```python colab={"base_uri": "https://localhost:8080/"} id="tZV48c8B-qfI" outputId="84f88ecc-221a-4ca0-822b-ef23410e56c2"
sweep_id = wandb.sweep(sweep_config, project="gretel-timeseries-sweep")

```

<!-- #region id="woB06tdu-qfI" -->
# Step 3️⃣. Run the Sweep agent

<!-- #endregion -->

```python id="cmAnsgcF-qfJ"
from gretel_client.projects import create_or_get_unique_project
from gretel_client.projects.models import read_model_config
from gretel_client.projects.jobs import Status

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        model_config = read_model_config("synthetics/default")

        model_config["models"][0]["synthetics"]["params"]["epochs"] = config["epochs"]
        model_config["models"][0]["synthetics"]["params"]["learning_rate"] = config[
            "learning_rate"
        ]
        model_config["models"][0]["synthetics"]["params"]["vocab_size"] = config[
            "vocab_size"
        ]
        model_config["models"][0]["synthetics"]["params"]["rnn_units"] = config[
            "rnn_units"
        ]
        model_config["models"][0]["synthetics"]["params"]["batch_size"] = config[
            "batch_size"
        ]

        project = create_or_get_unique_project(name="wandb-synthetic-data")
        model = project.create_model_obj(
            model_config=model_config, data_source="training_data.csv"
        )
        model.submit_cloud()

        # Log training accuracy to wandb
        for status_update in model.poll_logs_status():
            for update in status_update.logs:
                if "ctx" in update.keys():
                    acc = update["ctx"].get("accuracy")
                    loss = update["ctx"].get("loss")
                    epoch = update["ctx"].get("epoch")
                    ts = update["ctx"].get("ts")
                    if acc:
                        wandb.log(
                            {"accuracy": acc, "loss": loss, "time": ts, "epoch": epoch}
                        )

        # Log synthetic quality score and training time to wandb
        training_time = model.billing_details["total_time_seconds"]
        if model.status == Status.ERROR:
            wandb.log({"sqs": 0, "training_time": training_time})
        else:
            report = model.peek_report()
            sqs = report["synthetic_data_quality_score"]["score"]
            wandb.log({"sqs": sqs, "training_time": training_time})

```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["77c0409724e54db7a86d5eccbdb3767a", "2ed1c08b5c3c4f63a80289634197a101", "ccf77bafa7bd49b1b73b7fbf9639e722", "1d8007c479c1465384008ac7cf493333", "5d157c1bfbf342a8820517b50e016096", "35dca731f3e240718e77561064a2a1c6", "e5089b9bfeda4936a2b6ef9e77870c32", "fa3499d299294a16abe0e8f6b65d22d4", "3aad46f1732046e78edff7987d01e68b", "1bffbb4abed44cc6a549b49891a99ff6", "4f20ce3da20a4ad0a104d5ccfe140424", "d9179c6dc6554250ac4139a9a077634b", "d8621e1ec4244ec2870299f0deb89f55", "ccf6998b4b5e4989ab87780fed6a9583", "279e350d5c80430e82d53f837b88a81b", "b7c82c9310e64e4bb583fdc3b6e9fefb", "567f3862b3f34f76b64006ed413b3b76", "61fad0c920614a0eb9c1130c77571f34", "1c8fa865eaa448148f28ac7a4b4de719", "ecafb22ba1ac4083b444560d6473933c", "edd2a928b6d34d9390a8a6453a21e07a", "46a9229baf444c6bb8cdb1b38e8e4ae2", "2a7a4b48ebe148f1a50489c2975d669f", "fa707944d1ee463aa0888ebbe7e2c48d", "2635635e399b4782a3942643124a7981", "3aa82045833f4d8e8709837c36a6d0a2", "54b607dee9224bda9ee88e0d47b84271", "c7cd7a8be16c445ab408b2658256498a", "7a066e24870a4110af18f486fac077be", "2aeca6da04fa445ba8858a1b34308ef8", "931dccfd8f444f1baa0f45f6b6af26dc", "8d42dd3133bf4febb4e74e74705079f4", "918ccec115a64d9a911a2f05591c1654", "7a80f0e9a8bd4ab1a8201126049d543d", "d86c6116328149018ddd61956c6917e4", "f7e887fb86184f429a4437e737e9d193", "99adbe9dc1df48169f531dfcf43ad48f", "80df0ccdcdd74aaa9e2053ece6eb9811", "df1d65edce454a6fb6f9286cdb80d9f0", "bd93d39027ea4310a0e7cd1d5f16d04e", "be1a413133fb48c8b7e6519549e616f8", "e1c3ef18cdc443c8a588d3c58c5ee2d9", "a0e97256228c4fe98039e70df48b936e", "2a52b057330b4a60a691085514d45f93", "a8d18fae72b64c0e895606663ce520fc", "d31c0e8efccb4173a5fbcb8e9bacc200", "6a7ade9d99ab4cd8994d217ffc800f95", "c99e9b2b8170474280b1e4307df92fae", "04731fa250d64e869e171a2370934b01", "593ffc01875240ccb53c72540c874a04", "39c6af1e6ea44f13b17383b5fc6f4c62", "4545f10d48504e4ab0571f7ce8219585", "0425fd949f8d44a5bbb5d3d50f7b9fa5", "1fc8dbc7e31c407b93f1cfc36eebb771", "1d5c9a4223f7464d980e5f84568f887f", "75cc0b6dfb5549fa876617100acb964c", "b9fbb0f018c74b568cd112a67dd9f78d", "bb7d9c3f073a4a1286c4368ca848d836", "c826918e628f4fcf8183058d97343e5e", "80e6456fd859426b90469dd5796df685", "75b099645b6c4d0a847223402ef77484", "4475ffa0b67b4e24b9e785e52a86cee1", "79bb7452c66f4da0bdc3f6aff9da450a", "6e1c9eed32da4a7bb004ecc98a059d84", "b92a924f8bb44b299576494065a37dcb", "d13072ed8f7b48da907c87257b65d43a", "0e0440acb86b416699ae93871b24ad2c", "c1cc866a49124544a6a86050a986c185", "eb87ff040679483cbd95eb2b16e945fa", "d629a5b2b88c43aa8e920aa0b890255e", "09b9ee7b9dd6414391f434e39a7ab2de", "62bff84ea74344fab5748845da368fbf", "72c8d50c877f4688bc2bf591b37e0cef", "ae3361d5ed4d4e5a92fc5bad883a4c10", "61eaed752e1f4117aa40fa7ed5d0139a", "c47a0ab77c4145f39574f7569a9f2342", "1976f5c90b4d47b8a1e3df1fcbafed8e", "41531280d15043b58582185964d6795c", "310b640f5b214db4add5008a7c819d4f", "a07a6675e9594658a67c7893b7774e9b", "a8da494f1df84ab7afa8f12bd80fa4af", "f50298be1beb4b52ac0cf762bbff3eb5", "be01f4f03dd449ed87d28d1a9b479392", "7ba7df2f68004b359c47053bf4af9a35", "20da100ed4644086aac3d95de48e67f9", "864ae8dc09bc4a5c8a46cf93f9143b91", "9ce09048eb2f4842bdea431c61cc3bc5", "a509ac473f3949f0830496e270b0185f", "763ecc8e1a8b418db5520e7437a0100d", "ae80a99af7184d289ffe78f1056fe096", "d70a117bca8f4793a33616029d4e3230", "e857291f8e56409fa931cee0e92532ae", "93a7ea44dfa144709fa6e00bef634963", "8e34043538884ee192ee21cab3fb0f6b", "656d3535c832490c8a5d026a36fa2189", "96001b9772294a398e88e1d21d6754fc", "68d58d46452544ab84423605d1a87e9d", "7008c367522a44b5a0389c685a68f65b", "07c7890936b44ba88665847aebc24a1a", "f2bf9253cbe14446afa347d1e4e4bf0a", "e8afd879e3b3476ab5b7df48d92f2edb", "826d60f75faf47fcbf0e6e26623da09f", "fd7e07543bb04e0b80444bb94a0807fd", "8e532a93e2644bd9896e3cacb169695a", "008eeacc7fc6482985aecca02d9ee5ef", "f313fe3ccd564349a556ad3757660150", "112cb1ad3b0949979bbd6d4bc44d66e1", "f4df2a7dbdd947ddb8cc67c8a6053790", "a86e76033ad048d88e983f64ff185458", "b2151d8088e44d2f8c4965147149f001", "eeb684d7889e4e76a1c718ca76d2d283", "dade2ec737774ecda6ac90d548f4987b", "38d2655399d74dbfa10a4da45199bcaa", "3ac91cfcb3664712a93736f394c7dcac", "dd1990b5f1e646ceb26b2e5cc7ffb2d9", "03b32894b7074585863706d5e7b83489", "f82a04c1284840df9de6d9981ddf2b56", "b226a9c3ec6d4c25bfb206a90639ef40", "eaddbf6c479d463584d765865b130968", "9394e7ae226147078580c8c1177ea427", "94d9dee571234115b3f68e4edbca3128", "bb483844a494494c923191a515cd461c", "4cc22d52b0f9416a84d6999d6d7788b7", "6bf5876dbf1c496e8b30b4ff8a47580d", "5d2477fa65de4e71ba552b0bfa92d9b5", "d6f24627dc9f40a49041c1487c751873", "0a2060ad0559491eae9fe34298560c51", "f23b88b0b3a94405b444e4a0fae33fab", "3191d66298a54614adf0eb773d28f27a", "4434c8494efe4e7c94585467bf257330", "e79860aa85e64e158f83c31650f0ca25", "d6dc18af7c734adcb3e6eab39eca9c90", "455bc95f35ba46019310da968b4455c7", "4fdd0ec9d0c34231972c95b3e05fe085", "eaed336eac884ec48245c5a85353e930", "339f9f1f07dc45a08412fc7d7a7f81af", "49c8d33dbffe48869acc393e7e03a6bf", "7e3e7b1bcf3340d2b392f3f863682866", "c5426a8b6e1040b7884c15410c5f8135", "7b5e6264932742bd9b141dcdcd069b45", "ac0f93487d0c4024bb3010b569c61969", "f5ace8babbff44f4b581bf7c7d73b92f", "c3ea9cb880ef4e9eb0e83f6493b67134", "5f367a0197814272959e4e11dc0f7688", "d38c7b82ff074e0bb163de946a0b42b5", "cefed44d0ef74249af6449ddf3a812bc", "1e4a1750b5a042ffb4a35c5756998181", "5d3e0fbd28a2408fb8da6cea2d15913b", "e640f9d0b3c84b7e8ea361cd2903d92d", "251f43bf55f64f95884600b8383b709b", "7db896af54904bef8d9972ac0dde84c4", "c82a83162f704941a388f2edc5f4867b", "8bbc64aa3c764b13b5d204feebb8b74c", "b463e5c0ccef45b693b4bc9deb148fc7", "27ea9b282d27498cad33b6da208fa733", "60c55c75a42c46c4be4172f5049da4e2", "e8a281db781b42dcba347f846318cfe1", "2f3c5fbb50b14b7e9ef71127ca49ae82", "27157683fabc4c5298ed1ddf0cb51bd9", "9e6c2a2b2e254ee2b052becfa97de100"]} id="_M-emRHs-qfK" outputId="55657cd2-35ae-4c5c-fe12-fbab404e9993"
wandb.login()
wandb.agent(sweep_id, train, count=20)

```
