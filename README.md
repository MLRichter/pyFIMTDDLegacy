# pyFIMTDD (Legacy)
A implementation of the FIMT-DD regression-tree algorithm, based on the paper "Learning model trees from evolving data streams" 

see: https://pdfs.semanticscholar.org/7035/12732c5687212f3e71ddb632ee97713c7150.pdf

The implementation is not based on existing implementations.
The version is debugged and tested for one-dimensional Data and Labels.

# Files 

- pyFIMTDD.py           is the actual FIMTDD-algorithm.
- FIMTDD_LS.py          an attempt to improve FIMTDD by replacing the leafnode-approximators with adaptive filters. Effectiveness ist still not proven. Current results are approximately 5-10% reduction in the cumulative Loss.
- FIMTDD_evaluator.py   an eval-framework for optimizing parameters and testing the algorithms
