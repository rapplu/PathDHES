# PathGES: An Efficient and Secure Graph Encryption Scheme for Shortest Path Queries

This is the associated artifact for the paper "PathGES: An Efficient and Secure Graph Encryption Scheme for Shortest Path Queries" by Francesca Falzon, Esha Ghosh, Kenneth G. Paterson, and Roberto Tamassia.

**Important:** This repository implements several cryptographic primitives (used for research purposes) which should not be used in production.

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

To run experiments for the GKT scheme (Ghosh et al. AsiaCCS 2022) first create the following directories:
```
mkdir databases/GKT-databases
mkdir results/GKT-Results
```

Then run the command

```
python3 benchmark-gkt.py DATA NUM-QUERIES NUM-CORES
```
where DATA, NUM-QUERIES, and NUM-CORES are specified as before.


To clean the data obtained by running PathGES and calculate the averages partitioned by path lengths (as found in our paper) run the following additional command:

```
python3 clean-data.py DATA
```
where DATA once again specifies the reuslts from the corresponding dataset DATA to be cleaned.

Similarly, clean the data obtained by running GKT and calculate the averages partitioned by path lengths (as found in our paper) run the following:

```
python3 clean-data-gkt.py DATA
```

For the DES and EMM implementations we adapt the schemes found here: https://github.com/cloudsecuritygroup/ers
