// Copyright (c) 2014 Anonymized. All rights reserved.
//
// Code submitted as supplementary material for manuscript:
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// Do not redistribute.

// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Recurrent Neural Networks Language Model (RNNLM) toolbox

#ifndef __DependencyTreeRNN____RnnDependencyTreeLib__
#define __DependencyTreeRNN____RnnDependencyTreeLib__

#include "RnnLib.h"
#include "RnnTraining.h"
#include "CorpusUnrollsReader.h"

class RnnTreeLM : public RnnLMTraining {
public:
  
  /**
   * Constructor for training/testing the model
   */
  RnnTreeLM(const std::string &filename, bool doLoadModel, bool debugMode)
  // We load the RNN or not, depending on whether the model file is present
  // otherwise simply set its filename
  : RnnLMTraining(filename, doLoadModel, debugMode),
  // Parameters set by default (can be overriden when loading the model)
  m_typeOfDepLabels(0), m_labels(1) {
    std::cout << "RnnTreeLM\n";
  }
  
public:
  
  /**
   * Before learning the RNN model, we need to learn the vocabulary
   * from the corpus. Note that the word classes may have been initialized
   * beforehand using ReadClasses. Computes the unigram distribution
   * of words from a training file, assuming that the existing vocabulary
   * is empty.
   */
  bool LearnVocabularyFromTrainFile(int numClasses);
  
  /**
   * Import the vocabulary from a text file.
   */
  void ImportVocabularyFromFile(std::string &filename, int numClasses) {
    m_corpusTrain.ImportVocabulary(filename);
    m_corpusValidTest.ImportVocabulary(filename);
    AssignVocabularyFromCorpora(numClasses);
  }

  /**
   * Return the number of labels (features) used in the dependency parsing.
   */
  int GetLabelSize() const { return m_labels.GetVocabularySize(); }
  
  /**
   * Set the mode of the dependency labels:
   * 0: no dependency labels used
   * 1: dependency labels concatenated to the word
   * 0: dependency labels used as features in the feature vector
   */
  void SetDependencyLabelType(int type) {
    m_typeOfDepLabels = type;
  }

  /**
   * Set the minimum number of word occurrences
   */
  void SetMinWordOccurrence(int val) {
    m_corpusVocabulary.SetMinWordOccurrence(val);
    m_corpusTrain.SetMinWordOccurrence(val);
    m_corpusValidTest.SetMinWordOccurrence(val);
  }
  
  /**
   * Add a book to the training corpus
   */
  void AddBookTrain(const std::string &filename) {
    m_corpusVocabulary.AddBookFilename(filename);
    m_corpusTrain.AddBookFilename(filename);
  }
  
  /**
   * Add a book to the test/validation corpus
   */
  void AddBookTestValid(const std::string &filename) {
    m_corpusValidTest.AddBookFilename(filename);
  }
  
  /**
   * Function that trains the RNN on JSON trees
   * of dependency parse
   */
  bool TrainRnnModel();
  
  /**
   * Function that tests the RNN on JSON trees
   * of dependency parse
   */
  bool TestRnnModel(const std::string &testFile,
                    const std::string &featureFile,
                    std::vector<double> &sentenceScores,
                    double &logProbability,
                    double &perplexity,
                    double &entropy,
                    double &accuracy);

protected:

  // Corpora
  CorpusUnrolls m_corpusVocabulary;
  CorpusUnrolls m_corpusTrain;
  CorpusUnrolls m_corpusValidTest;
  
  // Label vocabulary representation (label -> index of the label)
  int m_typeOfDepLabels;
  
  // Label vocabulary hashtables
  Vocabulary m_labels;

  // Label vocabulary representation (label -> index of the label)
  std::unordered_map<std::string, int> m_mapLabel2Index;
  
  // Reset the vector of feature labels
  void ResetFeatureLabelVector(RnnState &state) const;
  
  // Update the vector of feature labels
  void UpdateFeatureLabelVector(int label, RnnState &state) const;

  // Assign the vocabulary from the corpora to the model,
  // and compute the word classes.
  bool AssignVocabularyFromCorpora(int numClasses);
};

#endif /* defined(__DependencyTreeRNN____RnnDependencyTreeLib__) */
