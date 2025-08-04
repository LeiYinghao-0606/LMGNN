# LMGNN: Long-Range Graph Data Modeling with Mamba Attention for Recommender Systems

This repository provides the official implementation for our AAAI 2026 paper : " LMGNN: Long-Range Graph Modeling for Recommender Systems ".


## Requirements
Pytorch:
- Python 3.9.21  
- PyTorch 2.2.2+cu118  
- causal-conv1d 1.4.0  
- mamba-ssm 2.2.2  
- numpy 1.22.4 

## Getting Started

python main.py

## Usage
1.Prepare Datasets
Please unzip the datasets into the designated data/ directory before running any scripts.

2.Directory Setup
Make sure to create the Models/ directory to store model checkpoints, and the History/ directory to save training and testing logs.

3.Training and Testing
Use the following command lines to start training and testing on the four datasets:

### Yelp
```bash
python main.py --data Yelp
```
### Tmall
```bash
python main.py --data Tmall
```
### Amazon-Books
```bash
python main.py --data Amazon-Books
```
### MovieLens-10M
```bash
python main.py --data ml-10m
```
