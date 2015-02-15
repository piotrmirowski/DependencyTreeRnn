// Copyright (c) 2014 Piotr Mirowski. All rights reserved.
//                    piotr.mirowski@computer.org
//
// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Recurrent Neural Networks Language Model (RNNLM) toolbox
//
// Recurrent neural network based statistical language modeling toolkitsize
// Version 0.3f
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
// Extensions from 0.3e to 0.3f version done at Microsoft Research
//
// This code implements the following paper:
//   Tomas Mikolov and Geoffrey Zweig
//   "Context Dependent Recurrent Neural Network Language Model"
//   Microsoft Research Technical Report MSR-TR-2012-92 July 27th, 2012
//   IEEE Conference on Spoken Language Technologies
//   http://research.microsoft.com/apps/pubs/default.aspx?id=176926
//
// Contributions: Piotr Mirowski (piotr.mirowski@computer.org)
//                Geoffrey Zweig (gzweig@microsoft.com)
//                Davide Di Gennaro (dadigenn@microsoft.com)
//                Vitaly Vazhnais (v-vitalv@microsoft.com)
//                Francesco Nidito (frnidito@microsoft.com)
//                Daniel Voinea (frnidito@microsoft.com)

/*
 This file is based on or incorporates material from the projects listed below (collectively, "Third Party Code").
 Microsoft is not the original author of the Third Party Code. The original copyright notice and the license under which Microsoft received such Third Party Code,
 are set forth below. Such licenses and notices are provided for informational purposes only. Microsoft, not the third party, licenses the Third Party Code to you
 under the terms set forth in the EULA for the Microsoft Product. Microsoft reserves all rights not expressly granted under this agreement, whether by implication,
 estoppel or otherwise.
 
 RNNLM 0.3e by Tomas Mikolov
 
 Provided for Informational Purposes Only
 
 BSD License
 All rights reserved.
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other
 materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __DependencyTreeRNN____RnnTraining__
#define __DependencyTreeRNN____RnnTraining__

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "CorpusWordReader.h"
#include "RnnLib.h"


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
  m_fileCorrectSentenceLabels("") {
  }
  
  void SetTrainFile(const std::string &str) {
    m_trainFile = str;
  }
  
  void SetValidFile(const std::string &str) {
    m_validationFile = str;
  }
  
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
  
  void SetUnkPenalty(double penalty) {
    m_logProbabilityPenaltyUnk = penalty;
  }
  
  void SetGradientCutoff(double newGradient) {
    m_gradientCutoff = newGradient;
  }
  
  void SetIndependent(bool newVal) {
    m_areSentencesIndependent = newVal;
  }
  
  void SetLearningRate(double newAlpha) {
    m_learningRate = newAlpha;
    m_initialLearningRate = newAlpha;
  }
  
  void SetRegularization(double newBeta) {
    m_regularizationRate = newBeta;
  }
  
  void SetMinImprovement(double newMinImprovement) {
    m_minLogProbaImprovement = newMinImprovement;
  }
  
  void SetDirectOrder(int newsize) {
    m_directConnectionOrder = newsize;
  }
  
  void SetNumStepsBPTT(int val) {
    m_numBpttSteps = val;
  }
  
  void SetBPTTBlock(int val) {
    m_bpttBlockSize = val;
  }
  
  void SetDebugMode(bool mode) {        m_debugMode = mode;
  }
  
  void SetFeatureGamma(double val) {
    m_featureGammaCoeff = val;
  }
  
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
  virtual bool LearnVocabularyFromTrainFile();
  
  /// <summary>
  /// Read the classes from a file in the following format:
  /// word [TAB] class_index
  /// where class index is between 0 and n-1 and there are n classes.
  /// </summary>
  bool ReadClasses(const std::string &filename);
  
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
  virtual double TestRnnModel(const std::string &testFile,
                              const std::string &featureFile,
                              int &wordCounter,
                              std::vector<double> &sentenceScores);
  
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
  /// Add a token (word or multi-word entity) to the vocabulary vector
  /// and store it in the map from word string to word index
  /// and in the map from word index to word string.
  /// </summary>
  int AddWordToVocabulary(const std::string& word);
  
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
  
protected:
  
  /// <summary>
  /// Are we in debug mode?
  /// </summary>
  bool m_debugMode;
  
  /// <summary>
  /// Word counter
  /// </summary>
  long m_wordCounter;
  
  /// <summary>
  /// Backup weights and state
  /// </summary>
  RnnWeights m_weightsBackup;
  RnnState m_stateBackup;
  
  /// <summary>
  /// Classification labels
  /// </summary>
  std::vector<int> m_correctSentenceLabels;
  
  /// <summary>
  /// File containing the correct classification labels
  /// </summary>
  std::string m_fileCorrectSentenceLabels;
};

#endif /* defined(__DependencyTreeRNN____RnnTraining__) */
