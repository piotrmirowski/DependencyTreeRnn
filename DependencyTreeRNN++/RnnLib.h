// Copyright (c) 2014 Anonymized. All rights reserved.
//
// Code submitted as supplementary material for manuscript:
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// Do not redistribute.

// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Recurrent Neural Networks Language Model (RNNLM) toolbox

#ifndef __DependencyTreeRNN____rnnlmlib__
#define __DependencyTreeRNN____rnnlmlib__

#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "RnnState.h"
#include "RnnWeights.h"
#include "CorpusWordReader.h"
#include "Vocabulary.h"


/// <summary>
/// Main class storing the RNN model
/// </summary>
class RnnLM
{
public:

  /// <summary>
  /// Constructor
  /// </summary>
  RnnLM(const std::string &filename,
        bool doLoadModel);

  /// <summary>
  /// Load the model.
  /// </summary>
  void LoadRnnModelFromFile();

  /// <summary>
  /// Return the number of words/entity tokens in the vocabulary.
  /// </summary>
  /// <returns>Integer number</returns>
  int GetVocabularySize() const { return m_vocab.GetVocabularySize(); }

  /**
   * Return the number of units in the input (word) layer.
   */
  int GetInputSize() const { return m_state.GetInputSize(); }

  /**
   * Return the number of units in the input (word) layer.
   */
  int GetHiddenSize() const { return m_state.GetHiddenSize(); }

  /**
   * Return the number of units in the optional hidden compression layer.
   */
  int GetCompressSize() const { return m_state.GetCompressSize(); }

  /**
   * Return the number of units in the feature (e.g., topic) layer.
   */
  int GetFeatureSize() const { return m_state.GetFeatureSize(); }

  /**
   * Return the number of units in the output layer.
   */
  int GetOutputSize() const { return m_state.GetOutputSize(); }

  /**
   * Return the number of direct connections between input words
   * and the output word (i.e., n-gram features)
   */
  int GetNumDirectConnection() const { return m_weights.GetNumDirectConnection(); }

  /**
   * Return the number of direct connections between input words
   * and the output word (i.e., n-gram features)
   */
  int GetOrderDirectConnection() const { return m_state.GetOrderDirectConnection(); }

  /**
   * Return the number of vocabulary classes. These are specified
   * at training time and can be frequency-based or rely on more
   * complex max-entropy features of the word bigrams.
   */
  int GetNumClasses() const { return m_weights.GetNumClasses(); }

protected:

  /// <summary>
  /// Exponentiates x.
  /// </summary>
  /// <returns>exp(x)</returns>
  double SafeExponentiate(double val) const
  {
    // for numerical stability
    val = (val > 50) ? 50 : ((val < -50) ? -50 : val);
    return (exp(val));
  }

  /// <summary>
  /// Exponentiates x in base 10.
  /// </summary>
  /// <returns>10^x</returns>
  double ExponentiateBase10(double num) const
  {
    return exp(num * 2.302585093);
  }

  /// <summary>
  /// Apply the logistic sigmoid function to x.
  /// </summary>
  /// <returns>1 / (1 + exp(-x))</returns>
  double LogisticSigmoid(double val) const
  {
    return (1 / (1 + SafeExponentiate(-val)));
  }

  /// <summary>
  /// Matrix-vector multiplication routine, somewhat accelerated using loop
  /// unrolling over 8 registers. Computes y <- y + A * x, (i.e. adds A * x to y)
  /// where A is of size N x M, x is of length M and y is of length N.
  /// The operation can done on a contiguous subset of indices
  /// i in [idxYFrom, idxYTo[ of vector y
  /// and on a contiguous subset of indices j in [idxXFrom, idxXTo[ of vector x.
  /// </summary>
  void MultiplyMatrixXvectorBlas(std::vector<double> &vectorY,
                                 std::vector<double> &vectorX,
                                 std::vector<double> &matrixA,
                                 int widthMatrix,
                                 int idxYFrom,
                                 int idxYTo) const;

public:

  /// <summary>
  /// Return the index of a word in the vocabulary, or -1 if OOV.
  /// </summary>
  int SearchWordInVocabulary(const std::string& word) const;

  /// <summary>
  /// Go to the next char delim when reading a file.
  /// </summary>
  bool GoToDelimiterInFile(int delim, FILE *fi) const;

  /// <summary>
  /// Function used to initialize the RNN model to the specified dimensions
  /// of the layers and weight vectors. This is done at construction
  /// of the RNN model object and also during training time (not at runtime).
  /// It is not thread safe yet because there is this file (m_featureMatrixFile)
  /// that contains the topic model for the words (LDA-style, see the paper),
  /// that is loaded by the function. It also modifies the vocabulary hash tables.
  /// </summary>
  bool InitializeRnnModel(int sizeInput,
                          int sizeHidden,
                          int sizeFeature,
                          int sizeClasses,
                          int sizeCompress,
                          long long sizeDirectConnection,
                          int orderDirectConnection);

  /// <summary>
  /// Erase the hidden layer state and the word history.
  /// Needed when processing sentences/queries in independent mode.
  /// Updates the RnnState object.
  /// </summary>
  void ResetHiddenRnnStateAndWordHistory(RnnState &state) const;
  void ResetHiddenRnnStateAndWordHistory(RnnState &state,
                                         RnnBptt &bpttState) const;

  /// <summary>
  /// Erases only the word history.
  /// Needed when processing sentences/queries in independent mode.
  /// Updates the RnnState object.
  /// </summary>
  void ResetWordHistory(RnnState &state) const;
  void ResetWordHistory(RnnState &state,
                        RnnBptt &bpttState) const;

  /// <summary>
  /// Forward-propagate the RNN through one full step, starting from
  /// the lastWord w(t) and the previous hidden state activation s(t-1),
  /// as well as optional feature vector f(t)
  /// and direct n-gram connections to the word history,
  /// computing the new hidden state activation s(t)
  /// s(t) = sigmoid(W * s(t-1) + U * w(t) + F * f(t))
  /// x = V * s(t) + G * f(t) + n-gram_connections
  /// y(t) = softmax_class(x) * softmax_word_given_class(x)
  /// Updates the RnnState object (but not the weights).
  /// </summary>
  void ForwardPropagateOneStep(int lastWord,
                               int word,
                               RnnState &state);

  /// <summary>
  /// Given a target word class, compute the conditional distribution
  /// of all words within that class. The hidden state activation s(t)
  /// is assumed to be already computed. Essentially, computes:
  /// x = V * s(t) + G * f(t) + n-gram_connections
  /// y(t) = softmax_class(x) * softmax_word_given_class(x)
  /// but for a specific targetClass.
  /// Updates the RnnState object (but not the weights).
  /// </summary>
  void ComputeRnnOutputsForGivenClass(const int targetClass,
                                      RnnState &state);

  /// <summary>
  /// Copies the hidden layer activation s(t) to the recurrent connections.
  /// That copy will become s(t-1) at the next call of ForwardPropagateOneStep
  /// </summary>
  void ForwardPropagateRecurrentConnectionOnly(RnnState &state) const;

  /// <summary>
  /// Shift the word history by one and update last word.
  /// </summary>
  void ForwardPropagateWordHistory(RnnState &state,
                                   int &lastWord,
                                   const int word) const;

  /// <summary>
  /// One way of having additional features to the RNN is to fit a topic
  /// model to the past history of words. This can be achieved in a simple
  /// way if such a topic matrix (words vs. topics) has been computed.
  /// The feature vector f(t) is then simply an autoregressive
  /// (exponentially decaying) function of the topic model vectors
  /// for each word in the history.
  /// This works well when processing sentence in English but might not
  /// be appropriate for short queries, since the topic feature
  /// will be continuously reset.
  /// </summary>
  void UpdateFeatureVectorUsingTopicModel(int word, RnnState &state) const;

  /// <summary>
  /// This is currently unused, and we might not use topic model features at all.
  /// The idea is to load a matrix of size W * T, where W is the number of words
  /// and T is the number of topics. Each word is embedding into a topic vector.
  /// The algorithm for word embedding can be Latent Dirichlet Allocation,
  /// Latent Semantic Indexing, DSSM, etc...
  /// It however assumes that the topic of the sentence changes with each word
  /// and is based on longer word history, which is more appropriate for
  /// long English sentences than for queries.
  /// The function that needs to be called at runtime or during training is
  /// UpdateFeatureVectorUsingTopicModel
  /// </summary>
  bool LoadTopicModelFeatureMatrix();

  // Simply copy the hidden activations and gradients, as well as
  // the word history, from one state object to another state object.
  void SaveHiddenRnnState(const RnnState &stateFrom,
                          RnnState &stateTo) const;

public:

  // Log-probability of unknown words
  double m_logProbabilityPenaltyUnk;

  // Vocabulary hashtables
  Vocabulary m_vocab;

  // State variable representing all the input/feature/hidden/output layer
  // activations of the RNN. This specific variable is just an initial
  // value that is created when the RNN model is loaded or initialized.
  // The training/testing functions do not modify it, simply make
  // a copy of it (convenient way to initialize the state vectors
  // of the right sizes).
  RnnState m_state;

  // The RNN model weights are stored in this object. Once loaded,
  // they will not be updated if the RNN is simply run on new data
  // (e.g., NextWord). Of course, the training algorithm will change them.
  RnnWeights m_weights;

  // These BPTT data are not used when the RNN model is run,
  // only during training, but it was easier to store them here.
  RnnBptt m_bpttVectors;

protected:

  /// <summary>
  /// Is the training file set?
  /// </summary>
  bool m_isTrainFileSet;

  /// <summary>
  /// Is the model loaded?
  /// </summary>
  bool m_isModelLoaded;

  /// <summary>
  /// Training and validation files
  /// </summary>
  std::string m_trainFile;
  std::string m_validationFile;

  /// <summary>
  /// RNN model file, version and type
  /// </summary>
  std::string m_rnnModelFile;
  int m_rnnModelVersion;

  /// <summary>
  /// Topic features
  /// </summary>
  std::string m_featureFile;
  std::string m_featureValidationFile;
  std::string m_featureMatrixFile;
  double m_featureGammaCoeff;
  int m_featureMatrixUsed;

  /// <summary>
  /// This is used for the second way how to add features
  /// into the RNN: only matrix W * T is specified,
  /// where W = number of words (m_vocabSize)
  /// and T = number of topics (m_featureSize)
  /// </summary>
  std::vector<double> m_featureMatrix;

  /// <summary>
  /// RNN model learning parameters. All this information will simply
  /// be loaded from the model file and not used when the RNN is run.
  /// </summary>
  double m_learningRate;
  double m_initialLearningRate;
  bool m_doStartReducingLearningRate;
  double m_regularizationRate;
  double m_minLogProbaImprovement;
  double m_gradientCutoff;
  int m_numBpttSteps;
  int m_bpttBlockSize;

  /// <summary>
  /// Information relative to the training of the RNN
  /// </summary>
  int m_iteration;
  long m_numTrainWords;
  long m_currentPosTrainFile;

  /// <summary>
  /// Information relative to the classes
  /// </summary>
  bool m_usesClassFile;

  /// <summary>
  /// Are the sentences independent?
  /// </summary>
  bool m_areSentencesIndependent;
};

#endif /* defined(__DependencyTreeRNN____rnnlmlib__) */
