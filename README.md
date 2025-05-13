# PathDHES: Directed Hypergraph Encryption Scheme for Shortest B-Path Queries

This repository is based on [ffalzon/ges-camera](https://github.com/ffalzon/ges-camera), a graph encryption scheme 
for private shortest path queries based on the paper "PathGES: An Efficient and Secure Graph Encryption Scheme for Shortest 
Path Queries" [1] by Francesca Falzon, Esha Ghosh, Kenneth G. Paterson, and Roberto Tamassia.
This respository extends its core functionalities to support directed 
hypergraph [2], [3] encryption for private shortest B-path and distance path queries. 
Hypergraphs are generalizations of standard graphs where edges become hyperedges that contain multiple vertices in its
and head instead of just one vertex.

[1] Francesca Falzon, Esha Ghosh, Kenneth G Paerson, and Roberto Tamassia. Pathges: An efficient and secure graph 
encryption scheme for shortest path queries. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and 
Communications Security, pages 4047–4061, 2024.

[2] Giorgio Gallo, Giustino Longo, Stefano Pallottino, and Sang Nguyen. Directed hypergraphs and applications. 
Discrete applied mathematics, 42(2-3):177–201, 1993.

[3] Lars Relund Nielsen, Daniele Pretolani, and K Andersen. A remark on the definition of a b-hyperpath. Department 
of Operations Research, University of Aarhus, Tech. Rep, 2001.

**Important:** This repository implements several cryptographic primitives (used for research purposes) which 
should not be used in production.

## PathDHES Functionality
If you wish to run a general B-Path query, just run the following:
```
python3 query-bpath.py DATA SOURCE TARGET SETUP-FLAG NUM-CORES
```
Similarly, for a Direct Path query, just run:
```
python3 query-directpath.py DATA SOURCE TARGET SETUP-FLAG NUM-CORES
```
Note that the hypergraph data is stored using zero-indexing, unlike most of the examples included in previous notebooks, 
so keep that in mind when choosing a SOURCE node and a TARGET node. Moreover, if running a query on a dataset you already 
use, make sure to make SETUP-FLAG false in order to avoid issues in how the files are stored (currently, it seems like 
they don't get overwritten).

## Detailed Usage

Our experiments assume prior installation of Python 3 via the `python3` command.
First clone the repository. A list of dependancies can be found in ``requirements.txt``. 

Before running the code, create the following directories in which the databases and results will be stored:
```
mkdir databases
mkdir databases/PathGES-databases
mkdir results
mkdir results/PathGES-Results
```

To run the experiments for PathGES, run the following commmand from the root directory of the repository. Our experiments can be carried out using the following datasets: ``Ca-GrQc``, ``InternetRouting``, ``email-EU-core``, ``facebook-combined``, ``p2p-Gnutella08``, ``p2p-Gnutella04``, ``p2p-Gnutella25``, ``cali``, and ``sbb``.

```
python3 benchmark.py DATA SETUP-FlAG NUM-QUERIES NUM-CORES
```

where DATA specifies the graph dataset to encrypt, SETUP-FlAG specifies whether to run setup-up experiments (0 = don't run setup experiments), and NUM-QUERIES and NUM-CORES are integers specifying the number of queries to test and the number of cores to use, respectively. Note that setup must be run before running query experiments at least once for each dataset, in order to encrypt the graph. Once the graph is encrypted the query experiments may be run without setup.

To run experiments for the GKT scheme (Ghosh et al. AsiaCCS 2022), first create the following directories:
```
mkdir databases/GKT-databases
mkdir results/GKT-Results
```

Then run the command
```
python3 benchmark-gkt.py DATA NUM-QUERIES NUM-CORES
```
where DATA, NUM-QUERIES, and NUM-CORES are specified as before.


To clean the data obtained by running PathGES and calculate the averages partitioned by path lengths (as found in our paper), run the following additional command:
```
python3 clean-data.py DATA
```
where DATA, once again, specifies the reuslts from the corresponding dataset DATA to be cleaned.

Similarly, clean the data obtained by running GKT and calculate the averages partitioned by path lengths (as found in our paper) run the following:
```
python3 clean-data-gkt.py DATA
```

For the DES and EMM implementations we adapt the schemes found here: https://github.com/cloudsecuritygroup/ers
