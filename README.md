# Modeling the Speedup of using Varying Number of Threads for a Neural Network Classification Algorithm
The problem I will be exploring is testing the training speed of a neural network classification algorithm on varying numbers of thread counts per CPU and varying sample sizes of data. Understanding how the training speed changes according to thread count and sample size may provide insight as to when it will be useful to introduce more threads. This may help a company when attempting to allocate a discrete amount of resources for computing.

One of the most popular modules/frameworks used to build neural networks is TensorFlow. This particular module also has the advantage of controlling either the number of threads or the number of GPU’s while training a model. Therefore, TensorFlow will be used due to its popularity and capabilities of parallelization.  

For a further look at this problem look at concepts_and_visualizations.ipynb for technical analysis (coding aspect) and pds_final_report.doc for more of a qualitative perspective
