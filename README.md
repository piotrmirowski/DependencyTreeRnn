# DependencyTreeRnn
Dependency tree-based RNN

# Installation
1. Modify the path to the BLAS header (cblas.h) file, i.e., $BLASINCLUDE
   and the BLAS path, i.e., $BLASFLAGS, in file Makefile.
   Alternatively, make your own version of that Makefile.
2. Build the project:
```
> make
```
   or, using your custom Makefile:
```
> make -f YOUR_OWN_MAKEFILE
```
Note that the .o objects are stored in directory build/ and the executable is ./RnnDependencyTree
   
# Sample training script
Shell script train_rnn_holmes_debug.sh trains an RNN on a subset of a few books.
You need to modify the path to where the JSON book files are stored.

