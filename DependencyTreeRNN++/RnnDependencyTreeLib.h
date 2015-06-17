// Copyright (c) 2014-2015 Piotr Mirowski
//
// Piotr Mirowski, Andreas Vlachos
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// ACL 2015

// Based on code by Geoffrey Zweig and Tomas Mikolov
// for the Feature-Augmented RNN Tool Kit
// http://research.microsoft.com/en-us/projects/rnn/

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
    // If we use dependency labels, do not connect them to the outputs
    m_useFeatures2Output = false;
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
