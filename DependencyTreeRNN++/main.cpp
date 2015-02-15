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


int main(int argc, char *argv[])
{
  // Command line arguments
  CommandLineParser parser;
  parser.Register("debug", "bool", "Debugging level", "false");
  parser.Register("train", "string", "Training data file (pure text)");
  parser.Register("valid", "string", "Validation data file (pure text), using during training");
  parser.Register("test", "string", "Test data file (pure text)");
  parser.Register("sentence-labels", "string", "Validation/test sentence labels file (pure text)");
  parser.Register("path-json-books", "string", "Path to the book JSON files", "./");
  parser.Register("rnnlm", "string", "RNN language model file to use (save in training / read in test)");
  parser.Register("features", "string", "Potentially ginouromous auxiliary feature file for training/test data, with one vector per training/test word");
  parser.Register("features-valid", "string", "Potentially ginourmous auxiliary feature file for validation data, with one vector per validation word");
  parser.Register("feature-matrix", "string", "Topic model matrix with word representations (e.g., LDA, LSA, Word2Vec, etc...)");
  parser.Register("feature-labels-type", "int", "Dependency parsing labels: 0=none, 1=concatenate, 2=features");
  parser.Register("feature-gamma", "double", "Decay weight for features consisting of topic model vectors or label vectors", "0.9");
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
    ifstream checkStream(trainFilename);
    if (!checkStream) {
      cout << "ERROR: training data file not found!\n";
      return 1;
    }
  }
  // Search for validation file
  string validFilename;
  bool isValidDataSet = parser.Get("valid", validFilename);
  if (isValidDataSet) {
    ifstream checkStream(validFilename);
    if (!checkStream) {
      cout << "ERROR: training data file not found!\n";
      return 1;
    }
  }
  if (isTrainDataSet && !isValidDataSet) {
    cout << "ERROR: validation data file must be specified for training!\n";
    return 0;
  }
  // Search for test file
  string testFilename;
  bool isTestDataSet = parser.Get("test", testFilename);
  if (isTestDataSet) {
    ifstream checkStream(testFilename);
    if (!checkStream) {
      cout << "ERROR: test data file not found!\n";
      return 1;
    }
  }
  if (!isTestDataSet && !isTrainDataSet) {
    cout << "ERROR: training or testing file must be specified!\n";
    return 1;
  }
  // Search for test file
  string sentenceLabelsFilename;
  bool isSentenceLabelsSet = parser.Get("sentence-labels", sentenceLabelsFilename);
  if (isSentenceLabelsSet) {
    ifstream checkStream(sentenceLabelsFilename);
    if (!checkStream) {
      cout << "ERROR: sentence labels file not found!\n";
      return 1;
    }
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
    ifstream checkStream(jsonPathname);
    if (!checkStream) {
      cout << "ERROR: json book path not found!\n";
      return 1;
    }
  }
  
  // Search for training features file
  string featureTrainOrTestFilename;
  int numFeatures = 0;
  bool isFeatureTrainOrTestDataSet =
  parser.Get("features", featureTrainOrTestFilename);
  if (isFeatureTrainOrTestDataSet) {
    ifstream featureStream(featureTrainOrTestFilename);
    if (!featureStream) {
      cout << "ERROR: training feature file not found!\n";
      return 1;
    }
    // Read the number of features
    featureStream >> numFeatures;
  }
  // Search for validation/test features file
  string featureValidFilename;
  bool isFeatureValidFileSet =
  parser.Get("features-valid", featureValidFilename);
  if (isFeatureValidFileSet) {
    ifstream checkStream(featureValidFilename);
    if (!checkStream) {
      cout << "ERROR: valid feature data file not found!\n";
      return 1;
    }
  }
  // Search for feature matrix file
  string featureMatrixFilename;
  bool isFeatureMatrixSet =
  parser.Get("feature-matrix", featureMatrixFilename);
  if (isFeatureMatrixSet) {
    ifstream checkStream(featureMatrixFilename);
    if (!checkStream) {
      cout << "ERROR: feature matrix file not found!\n";
      return 1;
    }
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
    ifstream checkStream(classFilename);
    if (!checkStream) {
      cout << "ERROR: valid feature data file not found!\n";
      return 1;
    }
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
    cerr << "Number of direct connections must be positive; saw: " << sizeDirectNGramConnections << endl;
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
  
  if (isTrainDataSet && isRnnModelSet) {
    // Construct the RNN object, setting the filename, without loading anything
    //RnnLMTraining model(rnnModelFilename, loadModel, debugMode);
    RnnTreeLM model(rnnModelFilename, isRnnModelPresent, debugMode);
    
    // Add the book names to the training corpus
    ifstream trainFileStream(trainFilename);
    string filename;
    string pathname(jsonPathname);
    while (trainFileStream >> filename) {
      string fullname = pathname + filename;
      model.AddBookTrain(fullname);
    }
    
    // Add the book names to the validation corpus
    ifstream valid_file_stream(validFilename);
    while (valid_file_stream >> filename) {
      string fullname = pathname + filename;
      model.AddBookTestValid(fullname);
    }
    
    // Set the filenames
    model.SetTrainFile(trainFilename);
    model.SetValidFile(validFilename);
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

    // Set the sentence labels for validation or test
    model.SetSentenceLabelsFile(sentenceLabelsFilename);
    // Do we use custom classes?
    // In a weird way, we read the classes before initializing the RNN
    if (isClassFileSet) {
      model.ReadClasses(classFilename);
    }
    
    // Set the minimum number of word occurrence
    model.SetMinWordOccurrence(minWordOccurrence);
    // Extract the vocabulary from the training file
    model.LearnVocabularyFromTrainFile();

    int sizeVocabulary = static_cast<int>(model.m_vocabularyStorage.size());
    int sizeVocabLabels =
    (featureDepLabelsType == 2) ? static_cast<int>(model.GetLabelSize()) : 0;
    if (!isRnnModelPresent) {
      // Initialize the model...
      model.InitializeRnnModel(sizeVocabulary,
                               sizeHiddenLayer,
                               sizeCompressionLayer,
                               sizeVocabulary + numClasses,
                               sizeVocabLabels,
                               sizeDirectNGramConnections,
                               model.m_state,
                               model.m_weights,
                               model.m_bpttVectors);
      // Set the direct connections n-gram order
      model.SetDirectOrder(orderDirectNGramConnections);
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
      assert(model.GetNumDirectConnections() == sizeDirectNGramConnections);
      // One thing needs to be done: assign the words to classes!
      model.AssignWordsToClasses();
      model.LoadRnnModelFromFile();
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
    //model.SaveRnnModelToFile();
    model.TrainRnnModel();
  }
  
  // Test the RNN on the dataset
  if (isTestDataSet && isRnnModelSet) {
    RnnLMTraining model(rnnModelFilename, true, debugMode);
    // One thing needs to be done: assign the words to classes!
    model.AssignWordsToClasses();
    
    // Test the RNN on the test data
    int testWordCount = 0;
    vector<double> sentenceScores;
    model.TestRnnModel(testFilename,
                       featureTrainOrTestFilename,
                       testWordCount,
                       sentenceScores);
  }
  
  return 0;
}
