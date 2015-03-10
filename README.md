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

1. Parameters relative to the dataset:
  * **train** (string) Training data file (pure text)
  * **valid** (string) Validation data file (pure text), using during training
  * **test** (string) Test data file (pure text)
  * **sentence-labels** (string) Validation/test sentence labels file (pure text)
  * **path-json-books** (string) Path to the book JSON files

2. Parameters relative to the dependency labels
  * **feature-labels-type** (int) Dependency parsing labels:
    * 0 = none, use words only
    * 1 = concatenate label to word
    * 2 = use features in the feature vector, separate from words
  * **feature-gamma** (double) Decay weight for features consisting of label vectors [default: 0.9].
    * Values up to about 1.3 can be accepted (beyond that, the perplexity seems to become very large).
    * f(t) is a vector with D elements (e.g., D=44 types of dependency labels)
    * f(t) <- gamma * f(t-1), then set element at current label to 1
    * This value could be important for changing the weight given to dependency labels.
      * A value larger than 1 means that labels further past in time count more than those immediately in the past.
      * 1 means that there is no decay.
      * A value between 0 and 1 means that there is some decay.
      * 0 means that the decay is immediate.

3. RNN architecture parameters
  * **classes** (int) Number of word classes used in hierarchical softmax [default value: 200].
    * If vocabulary size if W, choose C around sqrt(W).
    * C = W means 1 class per word.
    * C = 1 means standard softmax.
  * **hidden** (int) Number of nodes in the hidden layer [default value: 100].
    * Try to go higher, perhaps up to 1000 (for 1M-word vocabulary).
    * Linear impact on speed.
  * **direct** (int) Size of max-entropy hash table storing direct n-gram connections, in millions of entries [default value: 0].
    * Basically, direct=1000 means that 1000x10000000 = 1G direct connections between context words and target word are considered.
    * However, it is not a proper hashtable (which would take too much memory) but a simple vector of 1G entries, with a hashing function that hashed into specific entries in that vector. Hash collisions are totally ignored.
    * Try using direct=1000 or even 2000 hashes if possible.
  * **direct-order** (int) Order of direct n-gram connections; 2 is like bigram max entropy features [default value: 3].
    * It works on tokens only, and values of 4 or beyond did not bring improvement in others LM tasks.

4. Training parameters
  * **alpha** (double) Initial learning rate during gradient descent [default: 0.1]
  * **beta** (double) L-2 norm regularization coefficient during gradient descent [default: 0.0000001]
  * **min-improvement** (double) Minimum improvement before learning rate decreases [default: 1.001]
  * **bptt** (int) Number of steps to propagate error back in time [default: 4]
  * **bptt-block** (int) Number of time steps after which the error is backpropagated through time [default: 10]

1. debug", "bool", "Debugging level", "false");
7. rnnlm", "string", "RNN language model file to use (save in training / read in test)");
8. features", "string", "Potentially ginouromous auxiliary feature file for training/test data, with one vector per training/test word");
9. features-valid", "string", "Potentially ginourmous auxiliary feature file for validation data, with one vector per validation word");
10. feature-matrix", "string", "Topic model matrix with word representations (e.g., LDA, LSA, Word2Vec, etc...)");
  parser.Register("class-file", "string", "File specifying the class of each word");
  parser.Register("gradient-cutoff", "double", "decay weight for features matrix", "15");
  parser.Register("independent", "bool", "Is each line in the training/testing file independent?", "true");
  parser.Register("compression", "int", "Number of nodes in the compression layer", "0");
  parser.Register("unk-penalty", "double", "Penalty to add to <unk> in rescoring; normalizes type vs. token distinction", "-11");
  parser.Register("min-word-occurrence", "int", "Mininum word occurrence to include word into vocabulary", "3");
