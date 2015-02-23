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

# Important hyperparameters

1. debug", "bool", "Debugging level", "false");
2. train", "string", "Training data file (pure text)");
3. valid", "string", "Validation data file (pure text), using during training");
4. test", "string", "Test data file (pure text)");
5. sentence-labels", "string", "Validation/test sentence labels file (pure text)");
6. path-json-books", "string", "Path to the book JSON files", "./");
7. rnnlm", "string", "RNN language model file to use (save in training / read in test)");
8. features", "string", "Potentially ginouromous auxiliary feature file for training/test data, with one vector per training/test word");
9. features-valid", "string", "Potentially ginourmous auxiliary feature file for validation data, with one vector per validation word");
10. feature-matrix", "string", "Topic model matrix with word representations (e.g., LDA, LSA, Word2Vec, etc...)");
11. feature-labels-type", "int", "Dependency parsing labels: 0=none, 1=concatenate, 2=features");
12. feature-gamma", "double", "Decay weight for features consisting of topic model vectors or label vectors", "0.9");
  parser.Register("class", "int", "Number of classes", "200");
  parser.Register("class-file", "string", "File specifying the class of each word");
  parser.Register("gradient-cutoff", "double", "decay weight for features matrix", "15");
  parser.Register("independent", "bool", "Is each line in the training/testing file independent?", "true");
  parser.Register("alpha", "double", "Initial learning rate during gradient descent", "0.1");
  parser.Register("beta", "double", "L-2 norm regularization coefficient during gradient descent", "0.0000001");
  parser.Register("min-improvement", "double", "Minimum improvement before learning rate decreases", "1.001");
  parser.Register("hidden", "int", "Number of nodes in the hidden layer", "100");
  parser.Register("compression", "int", "Number of nodes in the compression layer", "0");
  parser.Register("direct", "int", "Size of max-ent hash table storing direct n-gram connections, in millions of entries", "0");
  parser.Register("direct-order", "int", "Order of direct n-gram connections; 2 is like bigram max ent features", "3");
  parser.Register("bptt", "int", "Number of steps to propagate error back in time", "4");
  parser.Register("bptt-block", "int", "Number of time steps after which the error is backpropagated through time", "10");
  parser.Register("unk-penalty", "double", "Penalty to add to <unk> in rescoring; normalizes type vs. token distinction", "-11");
  parser.Register("min-word-occurrence", "int", "Mininum word occurrence to include word into vocabulary", "3");
