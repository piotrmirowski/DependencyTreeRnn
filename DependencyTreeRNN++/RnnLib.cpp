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
#include <algorithm>
#include <sstream>
#include <fstream>

#include <math.h>
#include <time.h>
#include <assert.h>
#include "RnnLib.h"
#include "CorpusWordReader.h"

using namespace std;


/// <summary>
/// Return the index of a word in the vocabulary, or -1 if OOV.
/// </summary>
int RnnLM::SearchWordInVocabulary(const std::string& word) const
{
    auto i = m_mapWord2Index.find(word);
    if (i == m_mapWord2Index.end())
    {
        return -1;
    }
    else
    {
        return (i->second);
    }
}


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
bool RnnLM::LoadTopicModelFeatureMatrix()
{
    size_t numTopics = 0;
    
    std::ifstream inf(m_featureMatrixFile);
    std::string line;
    
    std::vector<double> topicVector;
    const auto vocabSize = GetVocabularySize();
    
    while (std::getline(inf, line))
    {
        // The file has the following structure:
        // there is one line per word, of format:
        // word topic_1 topic_2 ... topic_k
        // where each topic value is a double.
        // We tokenize on spaces.
        std::istringstream os(line);
        
        // Get the current word
        std::string word;
        os >> word;
        
        // Read the doubles into a topic vector for the current word
        topicVector.clear();
        double x;
        while (os >> x)
            topicVector.push_back(x);
        
        // Update the number of topics and reallocate feature matrix
        if (numTopics == 0)
        {
            numTopics = topicVector.size();
            m_featureMatrix.assign(numTopics, 10000.0);
            m_featureMatrix.assign(vocabSize * numTopics, 0.0);
        }
        
        // Find the index of the word...
        int wordIndex = SearchWordInVocabulary(word);
        if (wordIndex < 0 || wordIndex >= vocabSize)
            continue;
        // ... and store the topic vector for that word
        for (int a = 0; a < topicVector.size(); a++)
            m_featureMatrix[wordIndex + a * vocabSize] = topicVector[a];
    }
    return true;
}


/// <summary>
/// Function used to initialize the RNN model to the specified dimensions
/// of the layers and weight vectors. This is done at construction
/// of the RNN model object and also during training time (not at runtime).
/// It is not thread safe yet because there is this file (m_featureMatrixFile)
/// that contains the topic model for the words (LDA-style, see the paper),
/// that is loaded by the function. It also modifies the vocabulary hash tables.
/// </summary>
bool RnnLM::InitializeRnnModel(int sizeInput,
                                int sizeHidden,
                                int sizeCompress,
                                int sizeOutput,
                                int sizeFeature,
                                long long sizeDirectConnection,
                                RnnState &state,
                                RnnWeights &weights,
                                RnnBptt &bptt)
{
    if (!m_featureMatrixFile.empty())
    {
        // feature matrix file was set
        m_featureMatrixUsed = 1;
    }
    
    // This is currently unused, and we might not use topic model features at all
    if (m_featureMatrixUsed)
    {
        LoadTopicModelFeatureMatrix();
    }
    
    // The input consists of words and of s(t-1)
    //sizeInput = GetVocabularySize() + sizeHidden;
    sizeInput = GetVocabularySize();
    // The output has softmax over words (in each class) and over classes
    sizeOutput = GetVocabularySize() + m_numOutputClasses;
    
    // Initialize the input/hidden/output/compression/feature layers (nodes) of the RNN
    state.WordHistory.assign(c_maxNGramOrder, 0);
    state.InputLayer.assign(sizeInput, 0.0);
    state.InputGradient.assign(sizeInput, 0.0);
    state.RecurrentLayer.assign(sizeHidden, 0.0);
    state.RecurrentGradient.assign(sizeHidden, 0.0);
    state.HiddenLayer.assign(sizeHidden, 0.0);
    state.HiddenGradient.assign(sizeHidden, 0.0);
    state.FeatureLayer.assign(sizeFeature, 0.0);
    state.FeatureGradient.assign(sizeFeature, 0.0);
    state.OutputLayer.assign(sizeOutput, 0.0);
    state.OutputGradient.assign(sizeOutput, 0.0);
    state.CompressLayer.assign(sizeCompress, 0.0);
    state.CompressGradient.assign(sizeCompress, 0.0);
    
    // Allocate the weights connecting those layers
    // (will be assigned random values later)
    weights.Input2Hidden.resize(sizeInput * sizeHidden);
    weights.Recurrent2Hidden.resize(sizeHidden * sizeHidden);
    weights.Features2Hidden.resize(sizeFeature * sizeHidden);
    weights.Features2Output.resize(sizeFeature * sizeOutput);
    if (sizeCompress == 0)
    {
        weights.Hidden2Output.resize(sizeHidden * sizeOutput);
    }
    else
    {
        // Add a compression layer between hidden nodes and outputs
        weights.Hidden2Output.resize(sizeHidden * sizeCompress);
        weights.Compress2Output.resize(sizeCompress * sizeOutput);
    }
    // Change that to proper normal distribution
    // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    RandomizeVector(weights.Input2Hidden);
    RandomizeVector(weights.Recurrent2Hidden);
    if (sizeFeature > 0)
    {
        RandomizeVector(weights.Features2Hidden);
        RandomizeVector(weights.Features2Output);
    }
    if (sizeCompress > 0)
    {
        RandomizeVector(weights.Compress2Output);
    }
    RandomizeVector(weights.Hidden2Output);
    
    // Initialize the direct n-gram connections
    weights.DirectNGram.assign(sizeDirectConnection, 0.0);
    
    // BPTT vectors (as in Back-Propagation Through Time) will be used during training
    if (m_numBpttSteps > 0)
    {
        bptt.History.assign(m_numBpttSteps + m_bpttBlockSize + 10, -1);
        bptt.HiddenLayer.assign((m_numBpttSteps + m_bpttBlockSize + 1) * sizeHidden, 0);
        bptt.HiddenGradient.assign((m_numBpttSteps + m_bpttBlockSize + 1) * sizeHidden, 0);
        bptt.FeatureLayer.assign((m_numBpttSteps + m_bpttBlockSize + 2) * sizeFeature, 0);
        bptt.FeatureGradient.assign((m_numBpttSteps + m_bpttBlockSize + 2) * sizeFeature, 0);
        bptt.WeightsInput2Hidden.assign(sizeInput * sizeHidden, 0);
        bptt.WeightsRecurrent2Hidden.assign(sizeHidden * sizeHidden, 0);
        bptt.WeightsFeature2Hidden.assign(sizeFeature * sizeHidden, 0);
    }
    
    double df = 0;
    double dd = 0;
    int a = 0;
    int b = 0;
    
    int sizeVocabulary = GetVocabularySize();
    
    if (m_usesClassFile)
    {
        // Custom-specified classes, provided in a file, were used
        // at training time. There is nothing to do at this point,
        // just copy the class index for each word.
        int cnum = -1;
        int last = -1;
        for (int i = 0; i < sizeVocabulary; i++)
        {
            if (m_vocabularyStorage[i].classIndex != last)
            {
                last = m_vocabularyStorage[i].classIndex;
                m_vocabularyStorage[i].classIndex = ++cnum;
            }
            else
            {
                m_vocabularyStorage[i].classIndex = cnum;
            }
            // Unused
            m_vocabularyStorage[i].prob = 0.0;
        }
    }
    else
    {
        // Frequency-based classes (povey-style)
        // Re-assign classes based on the sqrt(word_count / total_word_count)
        // so that the classes contain equal weight of word occurrences.
        for (int i = 0; i < sizeVocabulary; i++)
        {
            b += m_vocabularyStorage[i].cn;
        }
        for (int i = 0; i < sizeVocabulary; i++)
        {
            dd += sqrt(m_vocabularyStorage[i].cn/ (double)b);
        }
        for (int i = 0; i < sizeVocabulary; i++)
        {
            df += sqrt(m_vocabularyStorage[i].cn / (double)b)/dd;
            if (df > 1)
            {
                df = 1;
            }
            if (df > (a + 1) / (double)m_numOutputClasses)
            {
                m_vocabularyStorage[i].classIndex = a;
                if (a < m_numOutputClasses-1)
                {
                    a++;
                }
            }
            else
            {
                m_vocabularyStorage[i].classIndex = a;
            }
            // Unused
            m_vocabularyStorage[i].prob = 0.0;
        }
    }
    
    // Store which words are in which class,
    // using a vector (length number of classes) of vectors (num words in that class)
    m_classWords.resize(m_numOutputClasses);
    for (int i = 0; i < sizeVocabulary; i++)
    {
        // Assign each word into its class
        int wordClass = m_vocabularyStorage[i].classIndex;
        m_classWords[wordClass].push_back(i);
    }
    
    // Check that there is no empty class
    for (int i = 0; i < m_numOutputClasses; i++)
    {
        if (m_classWords[i].empty())
        {
            printf("ERROR: class %d has no members\n", i);
            return false;
        }
    }
    return true;
}


/// <summary>
/// Read a matrix of floats in text format
/// </summary>
void ReadTextMatrix(FILE *fi, int sizeIn, int sizeOut, vector<double> &vec)
{
    for (int b = 0; b < sizeOut; b++)
    {
        for (int a = 0; a < sizeIn; a++)
        {
            double val;
            fscanf(fi, "%lf", &val);
            vec[a + b * sizeIn] = val;
        }
    }
}


/// <summary>
/// Read a matrix of floats in binary format
/// </summary>
void ReadBinaryMatrix(FILE *fi, int sizeIn, int sizeOut, vector<double> &vec)
{
    if (sizeIn * sizeOut == 0)
    {
        return;
    }
    for (int idxOut = 0; idxOut < sizeOut; idxOut++)
    {
        for (int idxIn = 0; idxIn < sizeIn; idxIn++)
        {
            float val;
            fread(&val, 4, 1, fi);
            vec[idxIn + idxOut * sizeIn] = val;
        }
    }
}


/// <summary>
/// Read a vector of floats in text format
/// </summary>
void ReadTextVector(FILE *fi, long long size, vector<double> &vec)
{
    for (long long aa = 0; aa < size; aa++)
    {
        double val;
        fscanf(fi, "%lf", &val);
        vec[aa] = val;
    }
}


/// <summary>
/// Read a vector of floats in binary format
/// </summary>
void ReadBinaryVector(FILE *fi, long long size, vector<double> &vec)
{
    for (long long aa = 0; aa < size; aa++)
    {
        float val;
        fread(&val, 4, 1, fi);
        vec[aa] = val;
    }
}


/// <summary>
/// Go to the next char delim when reading a file
/// </summary>
bool RnnLM::GoToDelimiterInFile(int delim, FILE *fi) const
{
    int ch = 0;
    while (ch != delim)
    {
        ch = fgetc(fi);
        if (feof(fi))
        {
            printf("Unexpected end of file\n");
            return false;
        }
    }
    return true;
}


/// <summary>
/// Constructor
/// Comment by Francesco: "I suggest loading the NN with Mmap, please
/// work with Vitaly/Davide in order to do something sensible here.
/// It doesn't make very much sense to load that one piece at the time...
/// and it should really be just a matrix of values..."
/// </summary>
RnnLM::RnnLM(const string &filename,
             bool doLoadModel)
// Default penalty for unknown words
: m_logProbabilityPenaltyUnk(-11.0),
// Was the model initialized?
m_isTrainFileSet(false),
m_isModelLoaded(false),
// Filename and type
m_rnnModelFile(filename),
// Internal model file version
m_rnnModelVersion(13),
m_isRnnModelFileBinary(true),
// Vanilla RNN without topic features
m_featureGammaCoeff(0.9),
m_featureMatrixUsed(false),
// Default learning parameters
m_learningRate(0.1),
m_initialLearningRate(0.1),
m_doStartReducingLearningRate(false),
m_regularizationRate(0.0000001),
m_minLogProbaImprovement(1.001),
m_gradientCutoff(15.0),
// BPTT of order 4, every 10 words
m_numBpttSteps(5),
m_bpttBlockSize(10),
// How many epochs was the RNN trained on?
m_iteration(0),
m_numTrainWords(0),
m_currentPosTrainFile(0),
// 200 classes by default
m_numOutputClasses(200),
m_numOldClasses(0),
// Use of a class file?
m_usesClassFile(false),
// Vanilla RNN without direct n-gram connections
m_directConnectionOrder(0),
// Independent sentences (queries)
m_areSentencesIndependent(true)
{
    // Load the RNN model?
    if (doLoadModel)
    {
        printf("# Loading RNN model from %s...\n", m_rnnModelFile.c_str());
        char buffer[8192];
        
        FILE *fi = fopen(m_rnnModelFile.c_str(), "rb");
        if (fi == NULL)
        {
            throw new runtime_error("Did not find file " + m_rnnModelFile);
        }
        
        GoToDelimiterInFile(':', fi);
        int ver = 13;
        fscanf(fi, "%d", &ver);
        if (ver != m_rnnModelVersion)
        {
            throw new runtime_error("Unknown version of file " + m_rnnModelFile);
        }
        
        GoToDelimiterInFile(':', fi);
        int binValue = 0;
        fscanf(fi, "%d", &binValue);
        m_isRnnModelFileBinary = (binValue > 0);
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%s", buffer);
        if (!m_isTrainFileSet)
        {
            m_trainFile = buffer;
        }
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%s", buffer);
        m_validationFile = buffer;
        
        double lastLogProbability;
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%lf", &lastLogProbability);
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%d", &m_iteration);
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%ld", &m_currentPosTrainFile);
        
        double dummyLogProbability = 0.0;
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%lf", &dummyLogProbability);
        
        GoToDelimiterInFile(':', fi);
        // temp kept for backwards compatibility in reading models
        int anti_k;
        fscanf(fi, "%d", &anti_k);
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%ld", &m_numTrainWords);
        
        GoToDelimiterInFile(':', fi);
        int sizeInput;
        fscanf(fi, "%d", &sizeInput);
        
        GoToDelimiterInFile(':', fi);
        int sizeFeature;
        fscanf(fi, "%d", &sizeFeature);
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%d", &m_featureMatrixUsed);
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%lf", &m_featureGammaCoeff);
        
        GoToDelimiterInFile(':', fi);
        int sizeHidden;
        fscanf(fi, "%d", &sizeHidden);
        
        GoToDelimiterInFile(':', fi);
        int sizeCompress;
        fscanf(fi, "%d", &sizeCompress);
        
        GoToDelimiterInFile(':', fi);
        int sizeOutput;
        fscanf(fi, "%d", &sizeOutput);
        
        long long sizeDirectConnection = 0;
        if (ver > 5)
        {
            GoToDelimiterInFile(':', fi);
            fscanf(fi, "%lld", &sizeDirectConnection);
        }
        
        if (ver > 6)
        {
            GoToDelimiterInFile(':', fi);
            fscanf(fi, "%d", &m_directConnectionOrder);
        }
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%d", &m_numBpttSteps);
        
        if (ver > 4)
        {
            GoToDelimiterInFile(':', fi);
            fscanf(fi, "%d", &m_bpttBlockSize);
        }
        else
        {
            m_bpttBlockSize = 10;
        }
        
        GoToDelimiterInFile(':', fi);
        int sizeVocabulary = 0;
        fscanf(fi, "%d", &sizeVocabulary);
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%d", &m_numOutputClasses);
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%d", &m_numOldClasses);
        
        GoToDelimiterInFile(':', fi);
        int booleanVal = 0;
        fscanf(fi, "%d", &booleanVal);
        m_usesClassFile = (booleanVal > 0);
        
        GoToDelimiterInFile(':', fi);
        booleanVal = 0;
        fscanf(fi, "%d", &booleanVal);
        m_areSentencesIndependent = (booleanVal > 0);
        
        GoToDelimiterInFile(':', fi);
        double val;
        fscanf(fi, "%lf", &val);
        m_initialLearningRate = val;
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%lf", &val);
        m_learningRate = val;
        
        GoToDelimiterInFile(':', fi);
        fscanf(fi, "%d", &m_doStartReducingLearningRate);
        
        GoToDelimiterInFile(':', fi);
        
        // Read the vocabulary, stored in text format as following:
        // index_number count word_token class_number
        // There are tabs and spaces separating the 4 columns
        m_vocabularyStorage.resize(sizeVocabulary);
        for (int a = 0; a < sizeVocabulary; a++)
        {
            // Read the word index and the word count
            int wordIndex;
            int wordCount;
            fscanf(fi, "%d%d", &wordIndex, &wordCount);
            assert(wordIndex == a);
            m_vocabularyStorage[a].cn = wordCount;
            // Read the word token
            char buffer[2048] = {0};
            if (fscanf(fi, "%s", &buffer))
                m_vocabularyStorage[a].word = buffer;
            // Read the class index
            int classIndex;
            fscanf(fi, "%d", &classIndex);
            
            // Store in the vocabulary vector and in the two maps
            m_vocabularyStorage[a].classIndex = classIndex;
            m_mapWord2Index[m_vocabularyStorage[a].word] = wordIndex;
            m_mapIndex2Word[wordIndex] = m_vocabularyStorage[a].word;
            string word = m_vocabularyStorage[wordIndex].word;
        }
        
        // Allocate the RNN here
        int a = m_featureMatrixUsed;
        m_featureMatrixUsed = 0;
        // memory allocation here
        InitializeRnnModel(sizeInput,
                           sizeHidden,
                           sizeCompress,
                           sizeOutput,
                           sizeFeature,
                           sizeDirectConnection,
                           m_state,
                           m_weights,
                           m_bpttVectors);
        m_featureMatrixUsed = a;
        
        // Read the activations on the hidden layer
        if (!m_isRnnModelFileBinary)
        {
            GoToDelimiterInFile(':', fi);
            ReadTextVector(fi, sizeHidden, m_state.HiddenLayer);
        }
        else
        {
            fgetc(fi);
            //printf("BINARY Loading %d-dim hidden state...\n", sizeHidden);
            ReadBinaryVector(fi, sizeHidden, m_state.HiddenLayer);
        }
        
        if (!m_isRnnModelFileBinary)
        {
            // Read the weights of input -> hidden connections
            GoToDelimiterInFile(':', fi);
            ReadTextMatrix(fi, sizeInput, sizeHidden, m_weights.Input2Hidden);
            // Read the weights of recurrent hidden -> hidden connections
            GoToDelimiterInFile(':', fi);
            ReadTextMatrix(fi, sizeHidden, sizeHidden, m_weights.Recurrent2Hidden);
            // Read the weights of feature -> hidden connections
            GoToDelimiterInFile(':', fi);
            ReadTextMatrix(fi, sizeFeature, sizeHidden, m_weights.Features2Hidden);
            // Read the weights of feature -> output connections
            GoToDelimiterInFile(':', fi);
            ReadTextMatrix(fi, sizeFeature, sizeOutput, m_weights.Features2Output);
            if (sizeCompress == 0)
            {
                // Read the weights of hidden -> output connections
                GoToDelimiterInFile(':', fi);
                ReadTextMatrix(fi, sizeHidden, sizeOutput, m_weights.Hidden2Output);
            }
            else
            {
                // Read the weights of hidden -> compression connections
                GoToDelimiterInFile(':', fi);
                ReadTextMatrix(fi, sizeHidden, sizeCompress, m_weights.Hidden2Output);
                // Read the weights of compression -> output connections
                GoToDelimiterInFile(':', fi);
                ReadTextMatrix(fi, sizeCompress, sizeOutput, m_weights.Compress2Output);
            }
            // Read the direct connections
            GoToDelimiterInFile(':', fi);
            ReadTextVector(fi, sizeDirectConnection, m_weights.DirectNGram);
            // Read the feature matrix
            if (m_featureMatrixUsed)
            {
                m_featureMatrix.resize(sizeVocabulary * sizeFeature);
                GoToDelimiterInFile(':', fi);
                ReadTextMatrix(fi, sizeFeature, sizeVocabulary, m_featureMatrix);
            }
        }
        else
        {
            // Read the weights of input -> hidden connections
            ReadBinaryMatrix(fi, sizeInput, sizeHidden, m_weights.Input2Hidden);
            // Read the weights of recurrent hidden -> hidden connections
            ReadBinaryMatrix(fi, sizeHidden, sizeHidden, m_weights.Recurrent2Hidden);
            // Read the weights of feature -> hidden connections
            ReadBinaryMatrix(fi, sizeFeature, sizeHidden, m_weights.Features2Hidden);
            // Read the weights of feature -> output connections
            ReadBinaryMatrix(fi, sizeFeature, sizeOutput, m_weights.Features2Output);
            if (sizeCompress == 0)
            {
                // Read the weights of hidden -> output connections
                ReadBinaryMatrix(fi, sizeHidden, sizeOutput, m_weights.Hidden2Output);
            }
            else
            {
                // Read the weights of hidden -> compression connections
                ReadBinaryMatrix(fi, sizeHidden, sizeCompress, m_weights.Hidden2Output);
                // Read the weights of compression -> output connections
                ReadBinaryMatrix(fi, sizeCompress, sizeOutput, m_weights.Compress2Output);
            }
            if (sizeDirectConnection > 0)
            {
                // Read the direct connections
                ReadBinaryVector(fi, sizeDirectConnection, m_weights.DirectNGram);
            }
            // Read the feature matrix
            if (m_featureMatrixUsed)
            {
                ReadBinaryMatrix(fi, sizeFeature, sizeVocabulary, m_featureMatrix);
                m_featureMatrix.resize(GetVocabularySize() * sizeFeature);
            }
        }
        fclose(fi);
        
        // Reset the state of the RNN
        ResetHiddenRnnStateAndWordHistory(m_state, m_bpttVectors);
        
        m_isModelLoaded = true;
    }
}


/// <summary>
/// Erase the hidden layer state and the word history.
/// Needed when processing sentences/queries in independent mode.
/// Updates the RnnState object.
/// </summary>
void RnnLM::ResetHiddenRnnStateAndWordHistory(RnnState &state) const
{
    // Set hidden unit activations to 1.0
    state.HiddenLayer.assign(GetHiddenSize(), 1.0);
    // Copy the hidden layer to the input (i.e., recurrent connection)
    ForwardPropagateRecurrentConnectionOnly(state);
    // Reset the word history
    ResetWordHistory(state);
}
void RnnLM::ResetHiddenRnnStateAndWordHistory(RnnState &state,
                                               RnnBptt &bpttState) const
{
    // Set hidden unit activations to 1
    // Copy the hidden layer to the input (i.e., recurrent connection)
    // Reset the word history
    ResetHiddenRnnStateAndWordHistory(state);
    
    // Reset the BPTT history and hidden layer activation and gradient
    if (m_numBpttSteps > 0)
    {
        for (int a = 1; a < m_numBpttSteps + m_bpttBlockSize; a++)
        {
            bpttState.History[a] = 0;
        }
        int sizeHidden = GetHiddenSize();
        for (int a = m_numBpttSteps + m_bpttBlockSize - 1; a > 1; a--)
        {
            for (int b = 0; b < sizeHidden; b++)
            {
                bpttState.HiddenLayer[a * sizeHidden + b] = 0;
                bpttState.HiddenGradient[a * sizeHidden + b] = 0;
            }
        }
    }
}


/// <summary>
/// Erases only the word history.
/// Needed when processing sentences/queries in independent mode.
/// Updates the RnnState object.
/// </summary>
void RnnLM::ResetWordHistory(RnnState &state) const
{
    state.WordHistory.assign(c_maxNGramOrder, 0);
}
void RnnLM::ResetWordHistory(RnnState &state,
                              RnnBptt &bpttState) const
{
    // Reset the word history
    ResetWordHistory(state);
    // Reset the word history in the BPTT
    if (m_numBpttSteps > 0)
    {
        for (int a = 0; a < m_numBpttSteps+m_bpttBlockSize; a++)
        {
            bpttState.History[a] = 0;
        }
    }
}


/// <summary>
/// Simply copy the hidden activations and gradients, as well as
/// the word history, from one state object to another state object.
/// </summary>
void RnnLM::SaveHiddenRnnState(const RnnState &stateFrom,
                                RnnState &stateTo) const
{
    stateTo.HiddenLayer.resize(stateFrom.HiddenLayer.size());
    stateTo.HiddenGradient.resize(stateFrom.HiddenGradient.size());
    stateTo.CompressLayer.resize(stateFrom.CompressLayer.size());
    stateTo.CompressGradient.resize(stateFrom.CompressGradient.size());
    stateTo.WordHistory.resize(c_maxNGramOrder);
    stateTo.HiddenLayer = stateFrom.HiddenLayer;
    stateTo.HiddenGradient = stateFrom.HiddenGradient;
    stateTo.CompressLayer = stateFrom.CompressLayer;
    stateTo.CompressGradient = stateFrom.CompressGradient;
    stateTo.WordHistory = stateFrom.WordHistory;
}


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
void RnnLM::ForwardPropagateOneStep(int lastWord,
                                    int word,
                                    RnnState &state) const
{
    // Nothing to do when the word is OOV
    if (word == -1)
    {
        return;
    }
    
    // The previous word (lastWord) is the input w(t) to the RN
    if (lastWord != -1)
    {
        state.InputLayer[lastWord] = 1;
    }
    
    // Erase activations of the hidden s(t) and hidden compression c(t) layers
    int sizeHidden = GetHiddenSize();
    int sizeCompress = GetCompressSize();
    state.HiddenLayer.assign(sizeHidden, 0.0);
    state.CompressLayer.assign(sizeCompress, 0.0);
    
    // Forward-propagate s(t-1) -> s(t)
    // using recurrent connection,
    // from previous value s(t-1) of the hidden layer at time t-1
    // to the current value s(t) of the hidden layer at time t
    // Operation: s(t) <- W * s(t-1)
    // Note that s(t-1) was previously copied to the recurrent input layer.
    int sizeInput = GetInputSize();
    MultiplyMatrixXvector(state.HiddenLayer,
                          state.RecurrentLayer,
                          m_weights.Recurrent2Hidden,
                          sizeHidden,
                          0,
                          sizeHidden);
    
    // Forward-propagate w(t) -> s(t)
    // from the one-hot word representation w(t) at time t
    // to the hidden layer s(t) at time t
    // Operation: s(t) <- s(t) + U * w(t)
    // Note that we add to s(t) which is already non-zero.
    if (lastWord != -1)
    {
        for (int b = 0; b < sizeHidden; b++)
        {
            state.HiddenLayer[b] +=
                state.InputLayer[lastWord] * m_weights.Input2Hidden[lastWord + b * sizeInput];
        }
    }
    
    int sizeFeature = GetFeatureSize();
    if (sizeFeature > 0)
    {
        // Forward-propagate f(t) -> s(t)
        // from the feature vector f(t) at time t
        // to the hidden layer s(t) at time t
        // Operation: s(t) <- s(t) + F * f(t)
        // Note that we add to s(t) which is already non-zero.
        MultiplyMatrixXvector(state.HiddenLayer,
                              state.FeatureLayer,
                              m_weights.Features2Hidden,
                              sizeFeature,
                              0,
                              sizeHidden);
    }
    
    // Apply the sigmoid transfer function to the hidden values s(t)
    // At this point, we have computed: z = W * s(t-1) + U * w(t) + F * f(t)
    // Operation: 1 / (1 + exp(-z))
    // We obtain: s(t) = sigmoid(W * s(t-1) + U * w(t) + F * f(t))
    for (int a = 0; a < sizeHidden; a++)
    {
        state.HiddenLayer[a] = LogisticSigmoid(state.HiddenLayer[a]);
    }
    
    if (sizeCompress > 0)
    {
        // Forward-propagate s(t) -> c(t)
        // from the hidden layer s(t) at time t
        // to the second (compression) hidden layer c(t) at time t
        // Operation: C * s(t)
        // TODO: check where CompressLayer was reset (should be)
        MultiplyMatrixXvector(state.CompressLayer,
                              state.HiddenLayer,
                              m_weights.Hidden2Output,
                              sizeHidden,
                              0,
                              sizeCompress);
        // Apply the sigmoid transfer function to the hidden values c(t)
        // Operation: 1 / (1 + exp(-z))
        // We obtain: c(t) = sigmoid(C * s(t))
        for (int a = 0; a < sizeCompress; a++)
        {
            state.CompressLayer[a] = LogisticSigmoid(state.CompressLayer[a]);
        }
    }
    
    // Reset the output layer (segment that encodes the class probabilities)
    int sizeOutput = GetOutputSize();
    int sizeVocabulary = GetVocabularySize();
    for (int b = sizeVocabulary; b < sizeOutput; b++)
    {
        state.OutputLayer[b] = 0;
    }
    
    if (sizeCompress > 0)
    {
        // Forward-propagate c(t) -> y(t)
        // from the second hidden (compression) layer c(t) at time t
        // to the output layer y(t) at time t
        // Operation: y(t) <- V * c(t)
        // Note that this operation is done only on the class outputs,
        // not on the word vocabulary per class outputs
        MultiplyMatrixXvector(state.OutputLayer,
                              state.CompressLayer,
                              m_weights.Compress2Output,
                              sizeCompress,
                              sizeVocabulary,
                              sizeOutput);
    }
    else
    {
        // Forward-propagate s(t) -> y(t)
        // from the hidden layer s(t) at time t
        // to the output layer y(t) at time t
        // Operation: y(t) <- V * s(t)
        // Note that this operation is done only on the class outputs,
        // not on the word vocabulary per class outputs
        MultiplyMatrixXvector(state.OutputLayer,
                              state.HiddenLayer,
                              m_weights.Hidden2Output,
                              sizeHidden,
                              sizeVocabulary,
                              sizeOutput);
    }
    
    if (sizeFeature > 0)
    {
        // Forward-propagate f(t) -> y(t)
        // from the feature layer f(t) at time t
        // to the output layer y(t) at time t
        // Operation: y(t) <- y(t) + G * f(t)
        // Note that this operation is done only on the class outputs,
        // not on the word vocabulary per class outputs
        MultiplyMatrixXvector(state.OutputLayer,
                              state.FeatureLayer,
                              m_weights.Features2Output,
                              sizeFeature,
                              sizeVocabulary,
                              sizeOutput);
    }
    
    // Apply direct connections to classes
    // TODO: this is a horrible mess, but the problem is that models
    // trained with this weird hashing function would be incompatible
    // with models trained with a proper hash table (unordered_map),
    // possibly sorted by the n-gram frequency.
    // It would be nice to make that change (and perhaps retrain old models).
    int sizeDirectConnection = GetNumDirectConnections();
    if (sizeDirectConnection > 0)
    {
        // this will hold pointers to m_weightDataMain.weightsDirect
        // that contains hash parameters
        unsigned long long hash[c_maxNGramOrder];
        
        for (int a = 0; a < m_directConnectionOrder; a++)
        {
            hash[a] = 0;
        }
        for (int a = 0; a < m_directConnectionOrder; a++)
        {
            int b = 0;
            if (a > 0)
            {
                if (state.WordHistory[a-1] == -1)
                {
                    // if OOV was in history, do not use this N-gram feature and higher orders
                    break;
                }
            }
            hash[a] = c_Primes[0] * c_Primes[1];
            
            for (b = 1; b <= a; b++)
            {
                hash[a] += c_Primes[(a * c_Primes[b] + b) % c_PrimesSize] * (unsigned long long)(state.WordHistory[b-1] + 1);
                // update hash value based on words from the history
            }
            // make sure that starting hash index is in the first half
            // of m_weightDataMain.weightsDirect (second part is reserved for history->words features)
            hash[a] = hash[a] % (sizeDirectConnection/2);
        }
        
        for (int a = sizeVocabulary; a < sizeOutput; a++)
        {
            for (int b = 0; b < m_directConnectionOrder; b++)
            {
                if (hash[b])
                {
                    // apply current parameter and move to the next one
                    state.OutputLayer[a] += m_weights.DirectNGram[hash[b]];
                    hash[b]++;
                }
                else
                {
                    break;
                }
            }
        }
    }
    
    // Apply the softmax transfer function to the hidden values s(t)
    // At this point, we have computed: x = V * s(t) + G * f(t)
    // Operation: exp(x_v) / sum_v exp(x_v)
    // We obtain: y(t) = softmax(V * s(t) + G * f(t) + n-gram features)
    // Note that this softmax is computed here only for classes, not words
    double sum = 0.0;
    for (int a = sizeVocabulary; a < sizeOutput; a++)
    {
        double val = SafeExponentiate(state.OutputLayer[a]);
        sum += val;
        state.OutputLayer[a] = val;
    }
    for (int a = sizeVocabulary; a < sizeOutput; a++)
    {
        state.OutputLayer[a] /= sum;
    }
    
    // What is the target class of the desired word?
    int targetClass = m_vocabularyStorage[word].classIndex;
    
    // Now, we need to compute the softmax for the words in that target class
    // (this will update the state)
    ComputeRnnOutputsForGivenClass(targetClass, state);
}


/// <summary>
/// Given a target word class, compute the conditional distribution
/// of all words within that class. The hidden state activation s(t)
/// is assumed to be already computed. Essentially, computes:
/// x = V * s(t) + G * f(t) + n-gram_connections
/// y(t) = softmax_class(x) * softmax_word_given_class(x)
/// but for a specific targetClass.
/// Updates the RnnState object (but not the weights).
/// </summary>
void RnnLM::ComputeRnnOutputsForGivenClass(int targetClass,
                                            RnnState &state) const
{
    // How many words in that target class?
    const auto targetClassCount = static_cast<int>(m_classWords[targetClass].size());
    // At which index in output layer y(t) position do the words
    // of the target class start?
    const auto minIndexWithinClass = m_classWords[targetClass][0];
    const auto maxIndexWithinClass = minIndexWithinClass + targetClassCount;
    // The indexes are in range [minIndexWithinClass, maxIndexWithinClass[
    // THIS WILL WORK ONLY IF CLASSES ARE CONTINUALLY DEFINED IN VOCABULARY
    // (i.e., class 10 = words 11 12 13; not 11 12 16)
    
    // Reset the outputs in y(t) for that class
    for (int c = 0; c < targetClassCount; c++)
    {
        state.OutputLayer[m_classWords[targetClass][c]] = 0;
    }
    
    int sizeCompress = GetCompressSize();
    int sizeHidden = GetHiddenSize();
    if (sizeCompress > 0)
    {
        // Forward-propagate c(t) -> y(t)
        // from the second hidden (compression) layer c(t) at time t
        // to the output layer y(t) at time t
        // Operation: y(t) <- V * c(t)
        // Note that this operation is done only on the words
        // in the class-specific vocabulary
        MultiplyMatrixXvector(state.OutputLayer,
                              state.CompressLayer,
                              m_weights.Compress2Output,
                              sizeCompress,
                              minIndexWithinClass,
                              maxIndexWithinClass);
    }
    else
    {
        // Forward-propagate s(t) -> y(t)
        // from the hidden layer s(t) at time t
        // to the output layer y(t) at time t
        // Operation: y(t) <- V * s(t)
        // Note that this operation is done only on the words
        // in the class-specific vocabulary
        MultiplyMatrixXvector(state.OutputLayer,
                              state.HiddenLayer,
                              m_weights.Hidden2Output,
                              sizeHidden,
                              minIndexWithinClass,
                              maxIndexWithinClass);
    }
    
    int sizeFeature = GetFeatureSize();
    if (sizeFeature > 0)
    {
        // Forward-propagate f(t) -> y(t)
        // from the feature layer f(t) at time t
        // to the output layer y(t) at time t
        // Operation: y(t) <- y(t) + G * f(t)
        // Note that this operation is done only on the words
        // in the class-specific vocabulary
        MultiplyMatrixXvector(state.OutputLayer,
                              state.FeatureLayer,
                              m_weights.Features2Output,
                              sizeFeature,
                              minIndexWithinClass,
                              maxIndexWithinClass);
    }
    
    // Apply direct connections to words
    // TODO: clean-up that mess...
    int sizeDirectConnection = GetNumDirectConnections();
    if (sizeDirectConnection > 0)
    {
        unsigned long long hash[c_maxNGramOrder];
        
        for (int a = 0; a < m_directConnectionOrder; a++)
        {
            hash[a] = 0;
        }
        for (int a = 0; a < m_directConnectionOrder; a++)
        {
            int b = 0;
            if ((a > 0) && (state.WordHistory[a-1] == -1))
            {
                break;
            }
            hash[a] = c_Primes[0]*c_Primes[1]*(unsigned long long)(targetClass+1);
            
            for (b = 1; b <= a; b++)
            {
                hash[a] += c_Primes[(a*c_Primes[b]+b)%c_PrimesSize]*(unsigned long long)(state.WordHistory[b-1]+1);
            }
            hash[a] = (hash[a] % (sizeDirectConnection/2)) + (sizeDirectConnection)/2;
        }
        
        for (int c = 0; c < targetClassCount; c++)
        {
            int a = m_classWords[targetClass][c];
            
            for (int b = 0; b < m_directConnectionOrder; b++) if (hash[b])
            {
                state.OutputLayer[a] += m_weights.DirectNGram[hash[b]];
                hash[b]++;
                hash[b] = hash[b]%sizeDirectConnection;
            }
            else
            {
                break;
            }
        }
    }
    
    // Apply the softmax transfer function to the hidden values s(t)
    // At this point, we have computed: x = V * s(t) + G * f(t)
    // Operation: exp(x_v) / sum_v exp(x_v)
    // We obtain: y(t) = softmax(V * s(t) + G * f(t) + n-gram features)
    // Note that this operation is done only on the words
    // in the class-specific vocabulary
    double sum = 0;
    for (int c = 0; c < targetClassCount; c++)
    {
        int wordIndex = m_classWords[targetClass][c];
        double val = SafeExponentiate(state.OutputLayer[wordIndex]);
        sum += val;
        state.OutputLayer[wordIndex] = val;
    }
    for (int c = 0; c < targetClassCount; c++)
    {
        int wordIndex = m_classWords[targetClass][c];
        state.OutputLayer[wordIndex] /= sum;
    }
}


/// <summary>
/// Copies the hidden layer activation s(t) to the recurrent connections.
/// That copy will become s(t-1) at the next call of ForwardPropagateOneStep
/// </summary>
void RnnLM::ForwardPropagateRecurrentConnectionOnly(RnnState &state) const
{
    state.RecurrentLayer = state.HiddenLayer;
}


/// <summary>
/// Shift the word history by one and update last word.
/// </summary>
void RnnLM::ForwardPropagateWordHistory(RnnState &state,
                                         int &lastWord,
                                         const int word) const
{
    // Delete the previous activation of the input layer for lastWord
    if (lastWord != -1)
    {
        state.InputLayer[lastWord] = 0;
    }
    // Update lastWord
    lastWord = word;
    // Shift the word history
    for (int a = c_maxNGramOrder - 1; a > 0; a--)
    {
        state.WordHistory[a] = state.WordHistory[a-1];
    }
    state.WordHistory[0] = lastWord;
}


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
void RnnLM::UpdateFeatureVectorUsingTopicModel(int word,
                                               RnnState &state) const
{
    // Safety check
    if (word < 0)
    {
        return;
    }
    
    // Check if the features for this word were defined
    if (m_featureMatrix[word] >= 1000)
    {
        return;
    }
    
    int sizeFeature = GetFeatureSize();
    int sizeVocabulary = GetVocabularySize();
    if (m_areSentencesIndependent && (word == 0))
    {
        // Reset the feature vector at the beginning of each sentence
        state.FeatureLayer.assign(sizeFeature, 0);
    }
    
    // The feature vector f is updated using exponential decay:
    // f(t) = gamma * f(t-1) +  (1 - gamma) * Z_word
    // where Z_word is the topic model word representation for the word.
    double oneMinusGamma = (1 - m_featureGammaCoeff);
    for (int a = 0; a < sizeFeature; a++)
    {
        state.FeatureLayer[a] =
        state.FeatureLayer[a] * m_featureGammaCoeff
        + m_featureMatrix[a * sizeVocabulary + word] * oneMinusGamma;
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
void RnnLM::MultiplyMatrixXvector(vector<double> &vectorY,
                                  const vector<double> &vectorX,
                                  const vector<double> &matrixA,
                                  int widthMatrix,
                                  int idxYFrom,
                                  int idxYTo) const
{
    int b;
    for (b = 0; b < (idxYTo - idxYFrom) / 8; b++)
    {
        double val1 = 0;
        double val2 = 0;
        double val3 = 0;
        double val4 = 0;
        double val5 = 0;
        double val6 = 0;
        double val7 = 0;
        double val8 = 0;
        for (int a = 0; a < widthMatrix; a++)
        {
            double vectorXa = vectorX[a];
            val1 += vectorXa * matrixA[a + (b * 8 + idxYFrom + 0) * widthMatrix];
            val2 += vectorXa * matrixA[a + (b * 8 + idxYFrom + 1) * widthMatrix];
            val3 += vectorXa * matrixA[a + (b * 8 + idxYFrom + 2) * widthMatrix];
            val4 += vectorXa * matrixA[a + (b * 8 + idxYFrom + 3) * widthMatrix];
            val5 += vectorXa * matrixA[a + (b * 8 + idxYFrom + 4) * widthMatrix];
            val6 += vectorXa * matrixA[a + (b * 8 + idxYFrom + 5) * widthMatrix];
            val7 += vectorXa * matrixA[a + (b * 8 + idxYFrom + 6) * widthMatrix];
            val8 += vectorXa * matrixA[a + (b * 8 + idxYFrom + 7) * widthMatrix];
        }
        vectorY[b * 8 + idxYFrom + 0] += val1;
        vectorY[b * 8 + idxYFrom + 1] += val2;
        vectorY[b * 8 + idxYFrom + 2] += val3;
        vectorY[b * 8 + idxYFrom + 3] += val4;
        vectorY[b * 8 + idxYFrom + 4] += val5;
        vectorY[b * 8 + idxYFrom + 5] += val6;
        vectorY[b * 8 + idxYFrom + 6] += val7;
        vectorY[b * 8 + idxYFrom + 7] += val8;
    }
    // An "elegant" way to do modulo 8...
    for (b = b * 8; b < idxYTo - idxYFrom; b++)
    {
        for (int a = 0; a < widthMatrix; a++)
        {
            vectorY[b + idxYFrom] += vectorX[a] * matrixA[a + (b + idxYFrom) * widthMatrix];
        }
    }
}
