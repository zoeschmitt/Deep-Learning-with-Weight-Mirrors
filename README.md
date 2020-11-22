# Deep Learning with Weight Mirrors
##### Zoe Schmitt, Julian Sherlock

Using research and code done by Mohamed Akrout, Collin Wilson, Peter C. Humphreys, Timothy Lillicrap, and Douglas Tweed: [Deep Learning without Weight Transport](https://arxiv.org/pdf/1904.05391.pdf)

# Getting started


## First-time setup

If this is your first time running the code, follow these steps:

1. Run `script/up` to create a virtual environment `.venv` with the required packages
2. Activate the virtual environment by running `source .venv/bin/activate`

## Running experiments

```bash

python main.py --n_epochs=400 --size_hidden_layers 500 --batch_size=128 --learning_rate=0.05 --test_frequency=1

```
