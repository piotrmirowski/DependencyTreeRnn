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
#include "CorpusWordReader.h"


//#define USE_HASHTABLES


/// <summary>
/// Max n-gram order, used for word history and direct connections
/// from the word history to the word output
/// </summary>
const int c_maxNGramOrder = 20;


/// <summary>
/// Element of vocabulary
/// </summary>
struct VocabWord
{
    std::string word;
    double prob;
    int cn;
    int classIndex;
};


// Tomas Mikolov decided to implement hash tables and hash functions
// from scratch...
const unsigned int c_Primes[] = {108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
    407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
    782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
const unsigned int c_PrimesSize = sizeof(c_Primes)/sizeof(c_Primes[0]);

#ifdef USE_HASHTABLES
struct WordTripleKey
{
    int w1;
    int w2;
    int w3;
    
    WordTripleKey(int v1, int v2, int v3)
    : w1(v1), w2(v2), w3(v3) { }

    bool isValid() { return (w1 != -1) && (w2 != -1) && (w3 != -1); }

    bool operator==(const WordTripleKey &key) const
    {
        return ((w1 == key.w1) && (w2 == key.w2) && (w3 == key.w3));
    }
};


template <>
struct std::hash<WordTripleKey>
{
    unsigned long long operator()(const WordTripleKey& k) const
    {
        unsigned long long hash = c_Primes[0] * c_Primes[1] * k.w1;
        hash += c_Primes[(2*c_Primes[1]+1)%c_PrimesSize] * k.w2;
        hash += c_Primes[(2*c_Primes[2]+2)%c_PrimesSize] * k.w3;
        return hash;
    }
};


struct WordPairKey
{
    int w1;
    int w2;
    
    WordPairKey(int v1, int v2)
    : w1(v1), w2(v2) { }
    
    bool isValid() { return (w1 != -1) && (w2 != -1); }

    bool operator==(const WordPairKey &key) const
    {
        return ((w1 == key.w1) && (w2 == key.w2));
    }
};


template <>
struct std::hash<WordPairKey>
{
    unsigned long long operator()(const WordPairKey& k) const
    {
        unsigned long long hash = c_Primes[0] * c_Primes[1] * k.w1;
        hash += c_Primes[(2*c_Primes[1]+1)%c_PrimesSize] * k.w2;
        return hash;
    }
};
#endif


struct RnnWeights
{
    // Weights between input and hidden layer
    std::vector<double> Input2Hidden;
    // Weights between former hidden state and current hidden layer
    std::vector<double> Recurrent2Hidden;
    // weights between features and hidden layer
    std::vector<double> Features2Hidden;
    // Weights between features and output layer
    std::vector<double> Features2Output;
    // Weights between hidden and output layer (or hidden and compression if compression>0)
    std::vector<double> Hidden2Output;
    // Optional weights between compression and output layer
    std::vector<double> Compress2Output;
    // Direct parameters between input and output layer (similar to Maximum Entropy model parameters)
#ifdef USE_HASHTABLES
    std::unordered_map<WordTripleKey, float> DirectTriGram;
    std::unordered_map<WordPairKey, float> DirectBiGram;
    std::unordered_map<int, float> DirectUniGram;
#else
    std::vector<double> DirectNGram;
#endif
};


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
    /// Assign words in vocabulary to classes (for hierarchical softmax).
    /// </summary>
    bool AssignWordsToClasses();
    
    /// <summary>
    /// Return the number of words/entity tokens in the vocabulary.
    /// </summary>
    /// <returns>Integer number</returns>
    int GetVocabularySize() const
    {
        return static_cast<int>(m_vocabularyStorage.size());
    }
    
    /// <summary>
    /// Return the number of units in the input (word) layer.
    /// </summary>
    /// <returns>Integer number</returns>
    int GetInputSize() const
    {
        return static_cast<int>(m_state.InputLayer.size());
    }
    
    /// <summary>
    /// Return the number of units in the input (word) layer.
    /// </summary>
    /// <returns>Integer number</returns>
    int GetHiddenSize() const
    {
        return static_cast<int>(m_state.HiddenLayer.size());
    }
    
    /// <summary>
    /// Return the number of units in the optional hidden compression layer.
    /// </summary>
    /// <returns>Integer number</returns>
    int GetCompressSize() const
    {
        return static_cast<int>(m_state.CompressLayer.size());
    }
    
    /// <summary>
    /// Return the number of units in the feature (e.g., topic) layer.
    /// </summary>
    /// <returns>Integer number</returns>
    int GetFeatureSize() const
    {
        return static_cast<int>(m_state.FeatureLayer.size());
    }
    
    /// <summary>
    /// Return the number of units in the output layer.
    /// </summary>
    /// <returns>Integer number</returns>
    int GetOutputSize() const
    {
        return static_cast<int>(m_state.OutputLayer.size());
    }
    
    /// <summary>
    /// Return the number of direct connections between input words
    /// and the output word (i.e., n-gram features)
    /// </summary>
    /// <returns>Integer number</returns>
    int GetNumDirectConnections() const
    {
#ifdef USE_HASHTABLES
        return 1;
#else
        return static_cast<int>(m_weights.DirectNGram.size());
#endif
    }
    
    /// <summary>
    /// Return the number of vocabulary classes. These are specified
    /// at training time and can be frequency-based or rely on more
    /// complex max-entropy features of the word bigrams.
    /// </summary>
    /// <returns>Integer number</returns>
    int GetNumClasses() const
    {
        return m_numOutputClasses;
    }
    
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
    /// Random number generator.
    /// </summary>
    /// <returns>Double random number in range [min, max]</returns>
    double GenerateUniformRandomNumber(double min, double max) const
    {
        return rand() / ((double)RAND_MAX) * (max - min) + min;
    }
    
    /// <summary>
    /// Random number generator (approximate Gaussian distribution).
    /// </summary>
    /// <returns>Zero-mean random number, standard deviation 0.1</returns>
    double GenerateNormalRandomNumber() const
    {
        return (GenerateUniformRandomNumber(-0.1, 0.1)
                + GenerateUniformRandomNumber(-0.1, 0.1)
                + GenerateUniformRandomNumber(-0.1, 0.1));
    }
    
    /// <summary>
    /// Randomize a vector with small numbers.
    /// </summary>
    /// <returns>Zero-mean random number</returns>
    void RandomizeVector(std::vector<double> &vec) const
    {
        for (size_t k = 0; k < vec.size(); k++)
        {
            vec[k] = GenerateNormalRandomNumber();
        }
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
                            int sizeCompress,
                            int sizeOutput,
                            int sizeFeature,
                            long long sizeDirectConnection,
                            RnnState &state,
                            RnnWeights &weights,
                            RnnBptt &bptt);
    
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
    
    /// <summary>
    /// Simply copy the hidden activations and gradients, as well as
    /// the word history, from one state object to another state object.
    /// </summary>
    void SaveHiddenRnnState(const RnnState &stateFrom,
                            RnnState &stateTo) const;
    
public:
    
    /// <summary>
    /// Log-probability of unknown words
    /// </summary>
    double m_logProbabilityPenaltyUnk;
    
    /// <summary>
    /// Vocabulary representation (word -> index of the word)
    /// </summary>
    std::unordered_map<std::string, int> m_mapWord2Index;
    
    /// <summary>
    /// Inverse vocabulary representation (index of the word -> word)
    /// </summary>
    std::unordered_map<int, std::string> m_mapIndex2Word;
    
    /// <summary>
    /// Vocabulary storage
    /// </summary>
    std::vector<VocabWord> m_vocabularyStorage;
    
    /// <summary>
    /// Hash table enabling a look-up of the class of a word
    /// (word -> word class)
    /// </summary>
    std::unordered_map<std::string, int> m_mapWord2Class;
    
    /// <summary>
    /// State variable representing all the input/feature/hidden/output layer
    /// activations of the RNN. This specific variable is just an initial
    /// value that is created when the RNN model is loaded or initialized.
    /// The training/testing functions do not modify it, simply make
    /// a copy of it (convenient way to initialize the state vectors
    /// of the right sizes).
    /// </summary>
    RnnState m_state;
    
    /// <summary>
    /// The RNN model weights are stored in this object. Once loaded,
    /// they will not be updated if the RNN is simply run on new data
    /// (e.g., NextWord). Of course, the training algorithm will change them.
    /// </summary>
    RnnWeights m_weights;
    
    /// <summary>
    /// These BPTT data are not used when the RNN model is run,
    /// only during training, but it was easier to store them here.
    /// </summary>
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
    std::set<int> m_classes;
    int m_numOutputClasses;
    std::vector<std::vector<int> > m_classWords;
    bool m_usesClassFile;
    
    /// <summary>
    /// The order of the n-gram connections is loaded from the model file
    /// and does not change when the model is run.
    /// </summary>
    int m_directConnectionOrder;
    
    /// <summary>
    /// Are the sentences independent?
    /// </summary>
    bool m_areSentencesIndependent;
};

#endif /* defined(__DependencyTreeRNN____rnnlmlib__) */
