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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <vector>
#include <time.h>

#include "CommandLineParser.h"
#include "RnnDependencyTreeLib.h"
#include "RnnTraining.h"

using namespace std;


bool checkFile(string filename, string msg) {
  ifstream checkStream(filename);
  if (!checkStream) {
    cout << "ERROR: did not find " << msg << " file " << filename << "\n";
    return false;
  }
  return true;
}


int main(int argc, char *argv[]) {
  // Command line arguments
  CommandLineParser parser;
  parser.Register("debug", "bool",
                  "Debugging level", "false");
  parser.Register("train", "string",
                  "Training data file (pure text)");
  parser.Register("valid", "string",
                  "Validation data file (pure text), using during training");
  parser.Register("test", "string",
                  "Test data file (pure text)");
  parser.Register("sentence-labels", "string",
                  "Validation/test sentence labels file (pure text)");
  parser.Register("path-json-books", "string",
                  "Path to the book JSON files", "./");
  parser.Register("rnnlm", "string",
                  "RNN language model file to use (save in training / read in test)");
  parser.Register("vocab", "string",
                  "File with vocabulary (used by word dependency-based RNN)");
  parser.Register("feature-labels-type", "int",
                  "Dependency parsing labels: -1 = sequential data, 0 = tree data but no dependency label, 1 = tree data, concatenate dependency label, 2 = tree data, use labels as features",
                  "-1");
  parser.Register("feature-gamma", "double",
                  "Decay weight for features consisting of topic model vectors or label vectors", "0.9");
  parser.Register("features", "string",
                  "Potentially ginouromous auxiliary feature file for training/test data, with one vector per training/test word");
  parser.Register("features-valid", "string",
                  "Potentially ginourmous auxiliary feature file for validation data, with one vector per validation word");
  parser.Register("feature-matrix", "string",
                  "Topic model matrix with word representations (e.g., LDA, LSA, Word2Vec, etc...)");
  parser.Register("class", "int",
                  "Number of classes", "200");
  parser.Register("class-file", "string",
                  "File specifying the class of each word");
  parser.Register("gradient-cutoff", "double",
                  "decay weight for features matrix", "15");
  parser.Register("independent", "bool",
                  "Is each line in the training/testing file independent?", "true");
  parser.Register("alpha", "double",
                  "Initial learning rate during gradient descent", "0.1");
  parser.Register("beta", "double",
                  "L-2 norm regularization coefficient during gradient descent", "0.0000001");
  parser.Register("min-improvement", "double",
                  "Minimum improvement before learning rate decreases", "1.001");
  parser.Register("hidden", "int",
                  "Number of nodes in the hidden layer", "100");
  parser.Register("compression", "int",
                  "Number of nodes in the compression layer", "0");
  parser.Register("direct", "int",
                  "Size of max-ent hash table storing direct n-gram connections, in millions of entries", "0");
  parser.Register("direct-order", "int",
                  "Order of direct n-gram connections; 2 is like bigram max ent features", "3");
  parser.Register("bptt", "int",
                  "Number of steps to propagate error back in time", "4");
  parser.Register("bptt-block", "int",
                  "Number of time steps after which the error is backpropagated through time", "10");
  parser.Register("unk-penalty", "double",
                  "Penalty to add to <unk> in rescoring; normalizes type vs. token distinction", "-11");
  parser.Register("min-word-occurrence", "int",
                  "Mininum word occurrence to include word into vocabulary", "3");
  
  // Parse the command line arguments
  bool status = parser.Parse(argv, argc);
  if (!status) {
    return 1;
  }
  
  // Set debug mode
  bool debugMode = false;
  parser.Get("debug", debugMode);
  
  // Search for train file
  string trainFilename;
  bool isTrainDataSet = parser.Get("train", trainFilename);
  if (isTrainDataSet) {
    if (!checkFile(trainFilename, "training data")) { return 1; }
  }
  // Search for validation file
  string validFilename;
  bool isValidDataSet = parser.Get("valid", validFilename);
  if (isValidDataSet) {
    if (!checkFile(validFilename, "validation data")) { return 1; }
  }
  if (isTrainDataSet && !isValidDataSet) {
    cout << "ERROR: validation data file must be specified for training!\n";
    return 0;
  }
  // Search for test file
  string testFilename;
  bool isTestDataSet = parser.Get("test", testFilename);
  if (isTestDataSet) {
    if (!checkFile(testFilename, "test data")) { return 1; }
  }
  if (!isTestDataSet && !isTrainDataSet) {
    cout << "ERROR: training or testing file must be specified!\n";
    return 1;
  }
  // Search for file containing the sentence labels
  string sentenceLabelsFilename;
  bool isSentenceLabelsSet = parser.Get("sentence-labels", sentenceLabelsFilename);
  if (isSentenceLabelsSet) {
    if (!checkFile(sentenceLabelsFilename, "sentence labels")) { return 1; }
  }
  if (!isTestDataSet && !isTrainDataSet) {
    cout << "ERROR: training or testing file must be specified!\n";
    return 1;
  }
  // Search for the RNN model file
  string rnnModelFilename;
  bool isRnnModelSet = parser.Get("rnnlm", rnnModelFilename);
  if (!isRnnModelSet) {
    cout << "ERROR: RNN model file not specified\n";
    return 1;
  }
  bool isRnnModelPresent = false;
  ifstream checkStream(rnnModelFilename);
  if (checkStream) {
    cout << "RNN model file exists\n";
    isRnnModelPresent = true;
  }
  if (isRnnModelSet && isTestDataSet && !isRnnModelPresent) {
    cout << "ERROR: RNN model file not found!\n";
    return 1;
  }
  // Search for the JSON book files path
  string jsonPathname;
  bool isJsonPathSet = parser.Get("path-json-books", jsonPathname);
  if (isJsonPathSet) {
    if (!checkFile(jsonPathname, "JSON book path")) { return 1; }
  }
  // Search for file containing the vocabulary
  string vocabularyFilename;
  bool isVocabularySet = parser.Get("vocab", vocabularyFilename);
  if (isVocabularySet) {
    if (!checkFile(vocabularyFilename, "vocabulary")) { return 1; }
  }
  if (!isTestDataSet && !isTrainDataSet) {
    cout << "ERROR: training or testing file must be specified!\n";
    return 1;
  }

  // Search for training features file
  string featureTrainOrTestFilename;
  int numFeatures = 0;
  bool isFeatureTrainOrTestDataSet =
  parser.Get("features", featureTrainOrTestFilename);
  if (isFeatureTrainOrTestDataSet) {
    if (!checkFile(featureTrainOrTestFilename, "train feature")) { return 1; }
    // Read the number of features
    ifstream featureStream(featureTrainOrTestFilename);
    featureStream >> numFeatures;
  }
  // Search for validation/test features file
  string featureValidFilename;
  bool isFeatureValidFileSet =
  parser.Get("features-valid", featureValidFilename);
  if (isFeatureValidFileSet) {
    if (!checkFile(featureValidFilename, "valid/test feature")) { return 1; }
  }
  // Search for feature matrix file
  string featureMatrixFilename;
  bool isFeatureMatrixSet =
  parser.Get("feature-matrix", featureMatrixFilename);
  if (isFeatureMatrixSet) {
    if (!checkFile(featureMatrixFilename, "feature matrix")) { return 1; }
  }
  // Set feature gamma
  double featureGammaCoeff = 0.9;
  parser.Get("feature-gamma", featureGammaCoeff);
  
  // Set the type of dependency labels
  int featureDepLabelsType = 0;
  parser.Get("feature-labels-type", featureDepLabelsType);
  
  // Set class size parameter
  int numClasses = 200;
  parser.Get("class", numClasses);
  // Check for a class specification file
  string classFilename;
  bool isClassFileSet = parser.Get("class-file", classFilename);
  if (isClassFileSet) {
    if (!checkFile(classFilename, "class data")) { return 1; }
  }
  
  // Set gradient cutoff
  double gradientCutoff = 15;
  parser.Get("gradient-cutoff", gradientCutoff);
  // Set independent
  bool independent = true;
  parser.Get("independent", independent);
  // Set learning rate
  double startingLearningRate = 0.1;
  parser.Get("alpha", startingLearningRate);
  // Set regularization
  double regularization = 0.0000001;
  parser.Get("beta", regularization);
  // Set min improvement of the validation log-probability
  double minLogProbaImprovement = 1.01;
  parser.Get("min-improvement", minLogProbaImprovement);
  // Set hidden layer size
  int sizeHiddenLayer = 100;
  parser.Get("hidden", sizeHiddenLayer);
  // Set compression layer size
  int sizeCompressionLayer = 0;
  parser.Get("compression", sizeCompressionLayer);
  // Set number of hashes for direct connections
  int temp;
  parser.Get("direct", temp);
  long long sizeDirectNGramConnections = 0;
  sizeDirectNGramConnections = temp * 1000000;
  if (sizeDirectNGramConnections < 0) {
    cerr << "Number of direct connections must be positive; saw: "
    << sizeDirectNGramConnections << endl;
    return 1;
  }
  // Set order of direct connections
  int orderDirectNGramConnections = 3;
  parser.Get("direct-order", orderDirectNGramConnections);
  if ((orderDirectNGramConnections > c_maxNGramOrder) ||
      (orderDirectNGramConnections < 0)) {
    cerr << "Direct n-gram order must be within 0 and " << c_maxNGramOrder << endl;
    return 1;
  }
  // Set number of steps for Backpropagation Through Time
  // (we count the first step as one)
  int bptt = 4;
  parser.Get("bptt", bptt);
  bptt++;
  if (bptt < 1) {
    bptt = 1;
  }
  // Set bptt block
  int bpttBlock = 10;
  parser.Get("bptt-block", bpttBlock);
  if (bpttBlock < 1) {
    bpttBlock = 1;
  }
  // Penalty for <unk>
  double unkPenalty = -11;
  parser.Get("unk-penalty", unkPenalty);
  // Minimum word occurrence
  int minWordOccurrence = 3;
  parser.Get("min-word-occurrence", minWordOccurrence);
  
  if (isTrainDataSet && isRnnModelSet && (featureDepLabelsType < 0)) {
    // Construct the RNN object, setting the filename, without loading anything
    RnnLMTraining model(rnnModelFilename, isRnnModelPresent, debugMode);

    // Set the training and validation file names
    model.SetTrainFile(trainFilename);
    model.SetValidFile(validFilename);
    // Set the sentence labels for validation or test
    model.SetSentenceLabelsFile(sentenceLabelsFilename);

    // Set the filenames
    /*
     if (isFeatureTrainOrTestDataSet) {
     if (!isFeatureValidFileSet)
     {
     cout << "ERROR: both training and valid feature files need to be set\n";
     return 1;
     }
     model.SetFeatureTrainOrTestFile(featureTrainOrTestFilename);
     model.SetFeatureValidationFile(featureValidFilename);
     //model.SetFeatureSize(numFeatures);
     }
     
     // Instead of pre-computed features, do we use topic model (LDA, LSA)
     // or other (e.g., Word2Vec) word features?
     if (isFeatureMatrixSet) {
     model.SetFeatureMatrixFile(featureMatrixFilename);
     }
     */

    // Read the vocabulary and word classes
    if (isClassFileSet) {
      // Do we use custom classes?
      model.ReadClasses(classFilename);
    } else {
      // Set the minimum number of word occurrence
      model.SetMinWordOccurrence(minWordOccurrence);
      // Extract the vocabulary from the training file
      model.LearnVocabularyFromTrainFile(numClasses);
    }

    // Initialize the model...
    int sizeVocabulary = model.GetVocabularySize();
    if (!isRnnModelPresent) {
      model.InitializeRnnModel(sizeVocabulary,
                               sizeHiddenLayer,
                               0,
                               numClasses,
                               sizeCompressionLayer,
                               sizeDirectNGramConnections,
                               orderDirectNGramConnections);
      // Set the feature label decay (gamma) weight
      model.SetFeatureGamma(featureGammaCoeff);
    } else {
      // ... or check that the model specification corresponds to what is loaded
      assert(model.GetInputSize() == sizeVocabulary);
      assert(model.GetHiddenSize() == sizeHiddenLayer);
      assert(model.GetCompressSize() == sizeCompressionLayer);
      assert(model.GetOutputSize() == sizeVocabulary + numClasses);
      assert(model.GetFeatureSize() == 0);
      assert(model.GetNumDirectConnection() == sizeDirectNGramConnections);
      assert(model.GetOrderDirectConnection() == orderDirectNGramConnections);
    }

    // When the model's training is restarting, these learning parameters
    // are simply ignored
    if (!isRnnModelPresent) {
      model.SetLearningRate(startingLearningRate);
      model.SetGradientCutoff(gradientCutoff);
      model.SetRegularization(regularization);
      model.SetMinImprovement(minLogProbaImprovement);
      model.SetNumStepsBPTT(bptt);
      model.SetBPTTBlock(bpttBlock);
      model.SetIndependent(independent);
    }
    
    // Train the model
    model.TrainRnnModel();
  }
  
  if (isTrainDataSet && isRnnModelSet && (featureDepLabelsType >= 0)) {
    // Construct the RNN object, setting the filename, without loading anything
    RnnTreeLM model(rnnModelFilename, isRnnModelPresent, debugMode);

    // Add the book names to the training corpus
    model.SetTrainFile(trainFilename);
    ifstream trainFileStream(trainFilename);
    string filename;
    string pathname(jsonPathname);
    while (trainFileStream >> filename) {
      string fullname = pathname + filename;
      model.AddBookTrain(fullname);
    }

    // Add the book names to the validation corpus
    model.SetValidFile(validFilename);
    ifstream valid_file_stream(validFilename);
    while (valid_file_stream >> filename) {
      string fullname = pathname + filename;
      model.AddBookTestValid(fullname);
    }
    // Set the sentence labels for validation or test
    model.SetSentenceLabelsFile(sentenceLabelsFilename);

    // Read the vocabulary and word classes
    if (isClassFileSet) {
      // Do we use custom classes?
      model.ReadClasses(classFilename);
    } else {
      if (isVocabularySet) {
        model.ImportVocabularyFromFile(vocabularyFilename, numClasses);
      } else {
        // Set the minimum number of word occurrence
        model.SetMinWordOccurrence(minWordOccurrence);
        // Extract the vocabulary from the training file
        model.LearnVocabularyFromTrainFile(numClasses);
      }
    }

    // Initialize the model...
    int sizeVocabulary = model.GetVocabularySize();
    int sizeVocabLabels =
    (featureDepLabelsType == 2) ? model.GetLabelSize() : 0;
    if (!isRnnModelPresent) {
      model.InitializeRnnModel(sizeVocabulary,
                               sizeHiddenLayer,
                               sizeVocabLabels,
                               numClasses,
                               sizeCompressionLayer,
                               sizeDirectNGramConnections,
                               orderDirectNGramConnections);
      // Set the type of dependency labels
      model.SetDependencyLabelType(featureDepLabelsType);
      // Set the feature label decay (gamma) weight
      model.SetFeatureGamma(featureGammaCoeff);
    } else {
      // ... or check that the model specification corresponds to what is loaded
      assert(model.GetInputSize() == sizeVocabulary);
      assert(model.GetHiddenSize() == sizeHiddenLayer);
      assert(model.GetCompressSize() == sizeCompressionLayer);
      assert(model.GetOutputSize() == sizeVocabulary + numClasses);
      assert(model.GetFeatureSize() == sizeVocabLabels);
      assert(model.GetNumDirectConnection() == sizeDirectNGramConnections);
      assert(model.GetOrderDirectConnection() == orderDirectNGramConnections);
      // TODO: this needs to be stored in the model
      // Set the type of dependency labels
      model.SetDependencyLabelType(featureDepLabelsType);
      // Set the feature label decay (gamma) weight
      model.SetFeatureGamma(featureGammaCoeff);
    }

    // When the model's training is restarting, these learning parameters
    // are simply ignored
    if (!isRnnModelPresent) {
      model.SetLearningRate(startingLearningRate);
      model.SetGradientCutoff(gradientCutoff);
      model.SetRegularization(regularization);
      model.SetMinImprovement(minLogProbaImprovement);
      model.SetNumStepsBPTT(bptt);
      model.SetBPTTBlock(bpttBlock);
      model.SetIndependent(independent);
    }

    // Train the model
    model.TrainRnnModel();
  }

  // Test the RNN on the dataset using models trained on dependency parse trees
  if (isTestDataSet && isRnnModelSet && (featureDepLabelsType >= 0)) {
    RnnTreeLM model(rnnModelFilename, true, debugMode);

    // Read the vocabulary
    if (!isVocabularySet)
      cerr << "Need to specify vocabulary file\n";
    model.ImportVocabularyFromFile(vocabularyFilename, model.GetNumClasses());

    // Add the book names to the test corpus
    model.SetValidFile(testFilename);
    ifstream test_file_stream(testFilename);
    string filename;
    string pathname(jsonPathname);
    while (test_file_stream >> filename) {
      string fullname = pathname + filename;
      cout << fullname << "\n";
      model.AddBookTestValid(fullname);
    }
    // Set the sentence labels for validation or test
    model.SetSentenceLabelsFile(sentenceLabelsFilename);
    // Set the type of dependency labels
    model.SetDependencyLabelType(featureDepLabelsType);

    // Test the RNN on the test data
    vector<double> sentenceScores;
    double logProbability, perplexity, entropy, accuracy;
    model.TestRnnModel(testFilename,
                       featureTrainOrTestFilename,
                       sentenceScores,
                       logProbability,
                       perplexity,
                       entropy,
                       accuracy);
  }
  
  // Test the RNN on the dataset using models trained on sequential text
  if (isTestDataSet && isRnnModelSet && (featureDepLabelsType < 0)) {
    RnnLMTraining model(rnnModelFilename, true, debugMode);

    // Add the book names to the test corpus
    model.SetValidFile(testFilename);
    // Set the sentence labels for validation or test
    model.SetSentenceLabelsFile(sentenceLabelsFilename);

    // Test the RNN on the test data
    vector<double> sentenceScores;
    double logProbability, perplexity, entropy, accuracy;
    model.TestRnnModel(testFilename,
                       featureTrainOrTestFilename,
                       sentenceScores,
                       logProbability,
                       perplexity,
                       entropy,
                       accuracy);
  }

  return 0;
}
