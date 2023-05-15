# LearnSC
This repository is an implementation of the paper *LearnSC: An Efficient and Unified Learning-based Framework for Subgraph Counting Problem*

## Requirements
This implementation has been tested on Ubuntu 20.04 with Python 3.9

|Package|Version|
|---|---|
|igraph|0.10.2|
|networkx|2.6.3|
|numpy|1.21.2|
|pandas|1.3.5|
|torch|1.10.1+cu113|
|torch-geometric|2.0.3|
## Usage
Before running python scripts, please add the src dir to LD_LIBRARY_PATH environment variable:
```bash
export LD_LIBRARY_PATH=/path/to/LearnSC/src:$LD_LIBRARY_PATH
```

Run main.py to train the model:

```sh
python main.py
```

The program arguments are list in the following table.

|Item|Type|Description|
|---|---|---|
|--dataname|Value (str)|Name of data graph|
|--n-query-node|Value (str)|Query graph size ("all" to input all available sizes)|
|--input-size|Value (int)|Number of distinct labels|
|--model-feat|Value (int)|Dimension of representations in the model|
|--batch-size|Value (int)|Batch size in model training and testing|
|--device|Value (str)|Device used for the model|
|--no-direction-embed|Option|NOT using loss for representation directions in interaction|
|--no-length-embed|Option|NOT using loss for projection length in interaction|
|--no-query-decomposition|Option|NOT decomposing query graphs|
|--no-ndec|Option|Using NeurSC's data graph decomposition|

This is a sample command:
```sh
python main.py --dataname yeast --n-query-node 10 --input-size 72
```

The candidate filter used in this repository is from paper [Sun and Luo. In-Memory Subgraph Matching: an In-depth Study. SIGMOD'20] and [Wang et al. Neural Subgraph Counting with Wasserstein Estimator. SIGMOD'22].
