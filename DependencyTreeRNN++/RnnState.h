// Copyright (c) 2014 Anonymized. All rights reserved.
//
// Code submitted as supplementary material for manuscript:
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// Do not redistribute.

// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Recurrent Neural Networks Language Model (RNNLM) toolbox

#ifndef DependencyTreeRNN___RnnState_h
#define DependencyTreeRNN___RnnState_h

#include <vector>
#include <algorithm>


/**
 * Max n-gram order, used for word history and direct connections
 * from the word history to the word output
 */
const int c_maxNGramOrder = 20;


/**
 * State vectors in the RNN model, storing per-word and per-class activations
 */
class RnnState {
public:

  /**
   * Constructor
   */
  RnnState(int sizeVocabulary,
           int sizeHidden,
           int sizeFeature,
           int sizeClasses,
           int sizeCompress,
           long long sizeDirectConnection,
           int orderDirectConnection)
  : m_orderDirectConnection(orderDirectConnection) {
    int sizeInput = sizeVocabulary;
    int sizeOutput = sizeVocabulary + sizeClasses;
    WordHistory.assign(c_maxNGramOrder, 0);
    InputLayer.assign(sizeInput, 0.0);
    InputGradient.assign(sizeInput, 0.0);
    RecurrentLayer.assign(sizeHidden, 0.0);
    RecurrentGradient.assign(sizeHidden, 0.0);
    HiddenLayer.assign(sizeHidden, 0.0);
    HiddenGradient.assign(sizeHidden, 0.0);
    FeatureLayer.assign(sizeFeature, 0.0);
    FeatureGradient.assign(sizeFeature, 0.0);
    OutputLayer.assign(sizeOutput, 0.0);
    OutputGradient.assign(sizeOutput, 0.0);
    CompressLayer.assign(sizeCompress, 0.0);
    CompressGradient.assign(sizeCompress, 0.0);
  }

  // Input layer (i.e., words)
  std::vector<double> InputLayer;
  // Input feature layer (e.g., topics)
  std::vector<double> FeatureLayer;
  // Hidden layer at previous time step
  std::vector<double> RecurrentLayer;
  // Hidden layer
  std::vector<double> HiddenLayer;
  // Second (compression) hidden layer
  std::vector<double> CompressLayer;
  // Output layer
  std::vector<double> OutputLayer;

  // Gradient to the words in input layer
  std::vector<double> InputGradient;
  // Gradient to the features in input layer
  std::vector<double> FeatureGradient;
  // Gradient to the hidden state at previous time step
  std::vector<double> RecurrentGradient;
  // Gradient to the hidden layer
  std::vector<double> HiddenGradient;
  // Gradient to the second (compression) hidden layer
  std::vector<double> CompressGradient;
  // Gradient to the output layer
  std::vector<double> OutputGradient;

  // Word history
  std::vector<int> WordHistory;


  /**
   * Return the number of units in the input (word) layer.
   */
  int GetInputSize() const {
    return static_cast<int>(InputLayer.size());
  }


  /**
   * Return the number of units in the input (word) layer.
   */
  int GetHiddenSize() const {
    return static_cast<int>(HiddenLayer.size());
  }


  /**
   * Return the number of units in the optional hidden compression layer.
   */
  int GetCompressSize() const {
    return static_cast<int>(CompressLayer.size());
  }


  /**
   * Return the number of units in the feature (e.g., topic) layer.
   */
  int GetFeatureSize() const {
    return static_cast<int>(FeatureLayer.size());
  }


  /**
   * Return the number of units in the output layer.
   */
  int GetOutputSize() const {
    return static_cast<int>(OutputLayer.size());
  }


  /**
   * Return the number of units in the output layer.
   */
  int GetOrderDirectConnection() const { return m_orderDirectConnection; }

protected:
  int m_orderDirectConnection;
};


class RnnBptt {
public:

  /**
   * Constructor
   */
  RnnBptt(int sizeVocabulary, int sizeHidden, int sizeFeature,
          int numBpttSteps, int bpttBlockSize)
  : m_bpttSteps(numBpttSteps), m_bpttBlock(bpttBlockSize),
  m_sizeHidden(sizeHidden), m_sizeFeature(sizeFeature),
  m_steps(0) {
    Reset();
    WeightsInput2Hidden.assign(sizeVocabulary * sizeHidden, 0);
    WeightsRecurrent2Hidden.assign(sizeHidden * sizeHidden, 0);
    WeightsFeature2Hidden.assign(sizeFeature * sizeHidden, 0);
  }


  /**
   * Number of BPTT steps that can be considered
   */
  int NumSteps() { return m_steps; }


  /**
   * Reset the BPTT memory
   */
  void Reset() {
    m_steps = 0;
    History.assign(m_bpttSteps + m_bpttBlock + 10, -1);
    HiddenLayer.assign((m_bpttSteps + m_bpttBlock + 1) * m_sizeHidden, 0);
    HiddenGradient.assign((m_bpttSteps + m_bpttBlock + 1) * m_sizeHidden, 0);
    FeatureLayer.assign((m_bpttSteps + m_bpttBlock + 2) * m_sizeFeature, 0);
    FeatureGradient.assign((m_bpttSteps + m_bpttBlock + 2) * m_sizeFeature, 0);
  }


  /**
   * Shift the BPTT memory by one
   */
  void Shift(int lastWord) {
    // Shift memory needed for BPTT to next time step
    if (m_bpttSteps > 0) {
      // shift memory needed for bptt to next time step
      for (int a = m_bpttSteps + m_bpttBlock - 1; a > 0; a--) {
        History[a] = History[a - 1];
      }
      History[0] = lastWord;

      for (int a = m_bpttSteps + m_bpttBlock - 1; a > 0; a--) {
        for (int b = 0; b < m_sizeHidden; b++) {
          HiddenLayer[a * m_sizeHidden + b] =
          HiddenLayer[(a - 1) * m_sizeHidden + b];
          HiddenGradient[a * m_sizeHidden + b] =
          HiddenGradient[(a - 1) * m_sizeHidden + b];
        }
      }

      for (int a = m_bpttSteps + m_bpttBlock - 1; a > 0; a--) {
        for (int b = 0; b < m_sizeFeature; b++) {
          FeatureLayer[a * m_sizeFeature+b] =
          FeatureLayer[(a - 1) * m_sizeFeature+b];
        }
      }
    }
    // Keep track of the number of that can be considered for BPTT
    m_steps++;
    m_steps = std::min(m_steps, m_bpttSteps + m_bpttBlock);
  }


  // Word history
  std::vector<int> History;
  // History of hidden layer inputs
  std::vector<double> HiddenLayer;
  // History of gradients to the hidden layer
  std::vector<double> HiddenGradient;
  // History of feature inputs
  std::vector<double> FeatureLayer;
  // History of gradients to the feature layer
  std::vector<double> FeatureGradient;
  // Gradients to the weights, to be added to the SGD gradients
  std::vector<double> WeightsInput2Hidden;
  std::vector<double> WeightsRecurrent2Hidden;
  std::vector<double> WeightsFeature2Hidden;


protected:
  // Number of steps gradients are back-propagated through time
  int m_bpttSteps;
  // How many steps (words) do we wait between consecutive BPTT?
  int m_bpttBlock;
  // How many steps have been stored since the last reset?
  int m_steps;
  // Number of hidden nodes
  int m_sizeHidden;
  // Number of features
  int m_sizeFeature;
};

#endif
