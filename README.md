# GraphFuse: Hybrid Graph Representation Learning for Money Laundering Detection

**Author:** Marius Frija  
**Supervisors:** Dr. Kubilay Atasu, Halil Çağrı Bilgi  
**Institution:** Delft University of Technology, EEMCS Faculty  
**Date:** June 20, 2025

---

## 📝 About the Thesis

This repository accompanies the bachelor thesis titled:

**"GraphFuse: Hybrid Graph Representation Learning for Money Laundering Detection"**


### Abstract

Money laundering detection stands as one of the most important challenges in the anti-financial crime sector, given its grave repercussions on the financial industry. The evolving nature of fraud schemes and the increasing volume of financial transactions impose limitations on the detection capabilities of traditional anti-money laundering (AML) systems. In light of the recent breakthroughs in the field of graph machine learning, graph neural networks (GNNs) and graph transformers (GTs) have emerged as prominent solutions to these limitations, achieving remarkable performance in detecting complex and broad fraudulent patterns.

However, fusing the powerful characteristics of these classes of graph models into a unified framework for fraud detection has been little explored. In this paper, we address this gap by presenting **GraphFuse** — a hybrid graph representation learning model tailored for money laundering detection in financial transaction graphs. The novel edge centrality and transaction signature encodings offer GraphFuse a slight advantage over the best-performing GNN and GT models, improving upon the best GT baseline by 0.76 p.p. in F1 score.

Additionally, we introduce three variants of the Transformer-based component of GraphFuse, each with a different level of computational complexity. The competitive performance of GraphFuse is supported by extensive experiments on open-source, large-scale synthetic financial transactions datasets.

📄 [Click here to read the full paper](Final_Bachelor_Thesis_M.Frija.pdf)

---

## 📦 Setup

To set up the development environment:

1. **Create and activate a new Conda environment:**

```bash
conda env create -f env.yml
conda activate aml-env
```
2. **Install Pytorch and Pytorch Geometric:**
```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

3. **Install the project requirements:**
```bash
pip install -r requirements.txt
```


## Data

The data needed for the experiments can be found on [Kaggle](https://www.kaggle.com/datasets/ealtman2019 ibm-transactions-for-anti-money-laundering-aml/data). 