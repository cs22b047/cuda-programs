# How to run

Non graph algorithms do not require to read from an external file. so, simply run them like this:
```
nvcc program-you-want-to-run.cu -o outfile
./outfile
```
For graph algorithms I'm reading graphs in .egr format provided at https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/

You run the programs like this:
```
nvcc graph-algo.cc example-graph.egr -o outfile
./outfile
```

# Outputs

## Baruvka's MST

```
Graph loaded: 1048576 nodes, 4190208 edges
totalEdges: 4190208
numNodes: 1048576
CPU Weight: 2806989831
CPU Time: 248.323 ms
GPU Time: 7.8928 ms
GPU Weight: 2806989831
```

## BST

```

```


## Prefix Sum

```

```

## Vector Addition

```

```

