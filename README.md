# PathDHES: Directed Hypergraph Encryption Scheme for Shortest B-Path Queries

This repository is based on [ffalzon/ges-camera](https://github.com/ffalzon/ges-camera), a graph encryption scheme 
for private shortest path queries based on the paper "PathGES: An Efficient and Secure Graph Encryption Scheme for Shortest Path Queries" [1] by Francesca Falzon, Esha Ghosh, Kenneth G. Paterson, and Roberto Tamassia. This respository extends its core functionalities to support directed 
hypergraph [2], [3] encryption for private shortest B-path and distance path queries. 
Hypergraphs are generalizations of standard graphs where edges become hyperedges that contain multiple vertices in its and head instead of just one vertex.

[1] Francesca Falzon, Esha Ghosh, Kenneth G Paerson, and Roberto Tamassia. Pathges: An efficient and secure graph 
encryption scheme for shortest path queries. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and 
Communications Security, pages 4047–4061, 2024.

[2] Giorgio Gallo, Giustino Longo, Stefano Pallottino, and Sang Nguyen. Directed hypergraphs and applications. 
Discrete applied mathematics, 42(2-3):177–201, 1993.

[3] Lars Relund Nielsen, Daniele Pretolani, and K Andersen. A remark on the definition of a b-hyperpath. Department 
of Operations Research, University of Aarhus, Tech. Rep, 2001.

[4] Blender Foundation. Blender agent 327 barber-shop demo file. https://www.blender.org/download/demo-files/, 2017.

**Important:** This repository implements several cryptographic primitives (used for research purposes) which should not be used in production.

## PathDHES Functionality

Our experiments assume prior installation of Python 3 via the `python3` command.
First clone the repository. A list of dependancies can be found in ``requirements.txt``. 

Before running the code, create the following directories in which the databases will be stored:
```
mkdir databases
mkdir databases/PathHES-databases
mkdir databases/BPathHES-databases
```

If you wish to run a general B-Path query, just run the following:
```
python3 query-bpath.py DATA SOURCE TARGET SETUP-FLAG NUM-CORES
```
Similarly, for a Distance Path query, just run:
```
python3 query-directpath.py DATA SOURCE TARGET SETUP-FLAG NUM-CORES
```
Note that the hypergraph data is stored using zero-indexing, unlike most of the examples included in previous notebooks, 
so keep that in mind when choosing a SOURCE node and a TARGET node. Moreover, if running a query on a dataset you already 
use, make sure to make SETUP-FLAG false in order to avoid issues in how the files are stored (currently, it seems like 
they don't get overwritten).

Either query type works if you wish to run it on regular directed graphs. We do not currently have regular graph datasets in the repository, but if you wish to experiment with your own examples, make sure to format the text files which contain them appropriately: each line of the file should be of the form ``INPUT;OUTPUT;EDGE-WEIGHT;``. If you wish to experiment with your own hypergraph datasets, the format is similar: ``COMMA-SEPARATED-INPUTS;COMMA-SEPARATED-OUTPUTS;HYPEREDGE-WEIGHTS;``.

To run the experiments for PathDHES, run the following commmand from the root directory of the repository. Our experiments can be carried out using the following datasets: ``hypergraph-dummy``, ``simple-hypergraph``, and ``test-hypergraph``. The last dataset is the same as the one contained in [4].

For the DES and EMM implementations we adapt the schemes found here: https://github.com/cloudsecuritygroup/ers.
