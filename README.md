## Setup

- Create a new Conda environment, and activate.
```bash
conda env create -f env.yml
conda activate aml-env

```
- Install Pytorch and Pytorch Geometric
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

## Data

The data needed for the experiments can be found on [Kaggle](https://www.kaggle.com/datasets/ealtman2019 ibm-transactions-for-anti-money-laundering-aml/data).