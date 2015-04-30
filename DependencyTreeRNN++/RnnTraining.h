// Copyright (c) 2014 Anonymized. All rights reserved.
//
// Code submitted as supplementary material for manuscript:
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// Do not redistribute.

// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Recurrent Neural Networks Language Model (RNNLM) toolbox

#ifndef __DependencyTreeRNN____RnnTraining__
#define __DependencyTreeRNN____RnnTraining__

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "CorpusWordReader.h"
#include "Utils.h"
#include "RnnLib.h"
#include "RnnState.h"


/// <summary>
/// Main class training and testing the RNN model,
/// not supposed at all to run in a production online environment
/// (not thread-safe).
/// </summary>
class RnnLMTraining : public RnnLM {
public:
  
  /// <summary>
  /// Constructor for training the model
  /// </summary>
  RnnLMTraining(const std::string &filename, bool doLoadModel, bool debugMode)
  // We load the RNN or not, depending on whether the model file is present
  // otherwise simply set its filename
  : RnnLM(filename, doLoadModel),
  m_debugMode(debugMode),
  m_wordCounter(0),
  m_minWordOccurrences(5),
  m_oov(1),
  m_eof(-2),
  m_fileCorrectSentenceLabels("") {
    Log("RnnLMTraining: debug mode is " + ConvString(debugMode) + "\n");
  }
  
  void SetTrainFile(const std::string &str) { m_trainFile = str; }
  
  void SetValidFile(const std::string &str) { m_validationFile = str; }
  
  void SetSentenceLabelsFile(const std::string &str) {
    m_fileCorrectSentenceLabels = str;
  }
  
  void SetFeatureTrainOrTestFile(const std::string &str) {
    m_featureFile = str;
  }
  
  void SetFeatureValidationFile(const std::string &str) {
    m_featureValidationFile = str;
  }
  
  void SetFeatureMatrixFile(const std::string &str) {
    m_featureMatrixFile = str;
  }
  
  void SetUnkPenalty(double penalty) { m_logProbabilityPenaltyUnk = penalty; }
  
  void SetGradientCutoff(double newGradient) {
    m_gradientCutoff = newGradient;
  }
  
  void SetIndependent(bool newVal) { m_areSentencesIndependent = newVal; }

  void SetLearningRate(double newAlpha) {
    m_learningRate = newAlpha;
    m_initialLearningRate = newAlpha;
  }
  
  void SetRegularization(double newBeta) { m_regularizationRate = newBeta; }
  
  void SetMinImprovement(double newMinImprovement) {
    m_minLogProbaImprovement = newMinImprovement;
  }

  /**
   * (Re)set the number of steps of BPTT
   */
  void SetNumStepsBPTT(int val) {
    m_numBpttSteps = val;
    m_bpttVectors = RnnBptt(GetVocabularySize(), GetHiddenSize(),
                            GetFeatureSize(),
                            m_numBpttSteps, m_bpttBlockSize);
  }
  
  /**
   * (Re)set the number of steps/words when BPTT is called
   */
  void SetBPTTBlock(int val) {
    m_bpttBlockSize = val;
    m_bpttVectors = RnnBptt(GetVocabularySize(), GetHiddenSize(),
                            GetFeatureSize(),
                            m_numBpttSteps, m_bpttBlockSize);
  }
  
  void SetDebugMode(bool mode) { m_debugMode = mode; }
  
  void SetFeatureGamma(double val) { m_featureGammaCoeff = val; }
  
public:
  
  /// <summary>
  /// Main function to train the RNN model
  /// </summary>
  virtual bool TrainRnnModel();
  
  /// <summary>
  /// Before learning the RNN model, we need to learn the vocabulary
  /// from the corpus. Note that the word classes may have been initialized
  /// beforehand using ReadClasses. Computes the unigram distribution
  /// of words from a training file, assuming that the existing vocabulary
  /// is empty.
  /// </summary>
  virtual bool LearnVocabularyFromTrainFile(int numClasses);
  

  /**
   * Set the minimum number of word occurrences
   */
  virtual void SetMinWordOccurrence(int val) {
    m_minWordOccurrences = val;
  }

  /// <summary>
  /// Read the classes from a file in the following format:
  /// word [TAB] class_index
  /// where class index is between 0 and n-1 and there are n classes.
  /// </summary>
  bool ReadClasses(const std::string &filename) {
    m_usesClassFile = m_vocab.ReadClasses(filename);
    return m_usesClassFile;
  }
  
  /// <summary>
  /// Once we train the RNN model, it is nice to save it to a text or binary file
  /// </summary>
  bool SaveRnnModelToFile();
  
  /// <summary>
  /// Simply write the word projections/embeddings to a text file.
  /// </summary>
  void SaveWordEmbeddings(const std::string &filename);
  
  /// <summary>
  /// Main function to test the RNN model
  /// </summary>
  virtual bool TestRnnModel(const std::string &testFile,
                            const std::string &featureFile,
                            std::vector<double> &sentenceScores,
                            double &logProbability,
                            double &perplexity,
                            double &entropy,
                            double &accuracy);
  
  /// <summary>
  /// Load a file containing the classification labels
  /// </summary>
  void LoadCorrectSentenceLabels(const std::string &labelFile);
  
protected:
  
  /// <summary>
  /// Get the next token (word or multi-word entity) from a text file
  /// and return it as an integer in the vocabulary vector.
  /// Returns -1 for OOV words and -2 for end of file.
  /// </summary>
  int ReadWordIndexFromFile(WordReader &reader);
  
  /// <summary>
  /// Sort the vocabulary by decreasing count of words in the corpus
  /// (used for frequency-based word classes, where class 0 contains
  /// </s>, class 1 contains {the} or another, most frequent token,
  /// class 2 contains a few very frequent tokens, etc...
  /// </summary>
  void SortVocabularyByFrequency();
  
  /// <summary>
  /// Sort the words by class, in increasing class order
  /// (used when the classes are provided by an external tools,
  /// e.g., based on maximum entropy features on word bigrams)
  /// </summary>
  void SortVocabularyByClass();
  
  /// <summary>
  /// One step of backpropagation of the errors through the RNN
  /// (optionally, backpropagation through time, BPTT) and of gradient descent.
  /// </summary>
  void BackPropagateErrorsThenOneStepGradientDescent(int last_word, int word);
  
  /// <summary>
  /// Read the feature vector for the current word
  /// in the train/test/valid file and update the feature vector
  /// in the state
  /// TODO: convert to ifstream
  /// </summary>
  bool LoadFeatureVectorAtCurrentWord(FILE *f, RnnState &state);
  
  /// <summary>
  /// Compute the accuracy of selecting the top candidate (based on score)
  /// among n-best lists
  /// </summary>
  double AccuracyNBestList(std::vector<double> scores,
                           std::vector<int> &correctClasses) const;
  
  /// <summary>
  /// Cleans all activations and error vectors, in the input, hidden,
  /// compression, feature and output layers, and resets word history
  /// </summary>
  void ResetAllRnnActivations(RnnState &state) const;
  
  /// <summary>
  /// Matrix-vector multiplication routine, accelerated using BLAS.
  /// Computes x <- x + A' * y,
  /// i.e., the "inverse" operation to y = A * x (adding the result to x)
  /// where A is of size N x M, x is of length M and y is of length N.
  /// The operation can done on a contiguous subset of indices
  /// j in [idxYFrom, idxYTo[ of vector y.
  /// </summary>
  void GradientMatrixXvectorBlas(std::vector<double> &vectorX,
                                 std::vector<double> &vectorY,
                                 std::vector<double> &matrixA,
                                 int widthMatrix,
                                 int idxYFrom,
                                 int idxYTo) const;
  
  /// <summary>
  /// Matrix-matrix multiplication routine, accelerated using BLAS.
  /// Computes C <- alpha * A * B + beta * C.
  /// The operation can done on a contiguous subset of row indices
  /// j in [idxRowCFrom, idxRowCTo[ in matrix A and C.
  /// </summary>
  void MultiplyMatrixXmatrixBlas(std::vector<double> &matrixA,
                                 std::vector<double> &matrixB,
                                 std::vector<double> &matrixC,
                                 double alpha,
                                 double beta,
                                 int numRowsA,
                                 int numRowsB,
                                 int numColsC,
                                 int idxRowCFrom,
                                 int idxRowCTo) const;
  
  /// <summary>
  /// Matrix-matrix or vector-vector addition routine using BLAS.
  /// Computes Y <- alpha * X + beta * Y.
  /// </summary>
  void AddMatrixToMatrixBlas(std::vector<double> &matrixX,
                             std::vector<double> &matrixY,
                             double alpha,
                             double beta,
                             int numRows,
                             int numCols) const;
  
protected:
  
  // Are we in debug mode?
  bool m_debugMode;
  
  // Word counter
  long m_wordCounter;
  
  // Index of the OOV (<unk>) word
  int m_oov;

  // Index of the EOF token
  int m_eof;

  // Minimum number of word occurrences
  int m_minWordOccurrences;

  // Classification labels
  std::vector<int> m_correctSentenceLabels;
  
  // File containing the correct classification labels
  std::string m_fileCorrectSentenceLabels;
};

#endif /* defined(__DependencyTreeRNN____RnnTraining__) */
