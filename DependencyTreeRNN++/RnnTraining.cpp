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
#include <climits>

#include <math.h>
#include <time.h>
#include <assert.h>
#include "RnnLib.h"
#include "RnnState.h"
#include "RnnTraining.h"
#include "CorpusWordReader.h"
// Include BLAS
extern "C" {
#include <cblas.h>
}

using namespace std;


/// <summary>
/// Get the next token (word or multi-word entity) from a text file
/// and return it as an integer in the vocabulary vector.
/// Returns -1 for OOV words and -2 for end of file.
/// </summary>
int RnnLMTraining::ReadWordIndexFromFile(WordReader &reader)
{
  std::string word = reader.get_next();
  // We return -2 if end of file, not to confuse with -1 for OOV words
  int index = -2;
  if (!(word.empty())) {
    index = SearchWordInVocabulary(word);
  }
  return index;
}


/// <summary>
/// Add a token (word or multi-word entity) to the vocabulary vector
/// and store it in the map from word string to word index
/// and in the map from word index to word string.
/// </summary>
int RnnLMTraining::AddWordToVocabulary(const std::string& word)
{
  // Initialize the word index, count and probability to 0
  VocabWord w = VocabWord();
  w.word = word;
  w.prob = 0.0;
  w.cn = 1;
  int index = static_cast<int>(m_vocabularyStorage.size());
  m_vocabularyStorage.push_back(std::move(w));
  // We need to store the word - index pair in the hash table word -> index
  // but we will rewrite that map later after sorting the vocabulary by frequency
  m_mapWord2Index[word] = index;
  return (index);
}


/// <summary>
/// Sort the vocabulary by decreasing count of words in the corpus
/// (used for frequency-based word classes, where class 0 contains
/// </s>, class 1 contains {the} or another, most frequent token,
/// class 2 contains a few very frequent tokens, etc...
/// </summary>
bool OrderWordCounts(const VocabWord& a, const VocabWord& b) { return a.cn > b.cn; }
void RnnLMTraining::SortVocabularyByFrequency()
{
  std::sort(m_vocabularyStorage.begin(),
            m_vocabularyStorage.end(),
            OrderWordCounts);
}


/// <summary>
/// Sort the words by class, in increasing class order
/// (used when the classes are provided by an external tools,
/// e.g., based on maximum entropy features on word bigrams)
/// </summary>
bool OrderClassIndex(const VocabWord& a, const VocabWord& b) {
  return a.classIndex < b.classIndex;
}
void RnnLMTraining::SortVocabularyByClass()
{
  // Sort the words by class, in increasing class order
  std::sort(m_vocabularyStorage.begin(),
            m_vocabularyStorage.end(),
            OrderClassIndex);
}


/// <summary>
/// Before learning the RNN model, we need to learn the vocabulary
/// from the corpus. Note that the word classes may have been initialized
/// beforehand using ReadClasses. Computes the unigram distribution
/// of words from a training file, assuming that the existing vocabulary
/// is empty.
/// </summary>
bool RnnLMTraining::LearnVocabularyFromTrainFile()
{
  // Word reader file on the training file
  WordReader wr(m_trainFile);
  
  // We reinitialize the vocabulary vector,
  // and the word -> index map,
  // but not the word -> class map which may have been loaded by ReadClasses.
  // Note that the map word -> index will be rebuilt after sorting the vocabulary.
  m_vocabularyStorage.clear();
  m_mapWord2Index.clear();
  
  // The first word needs to be end-of-sentence
  AddWordToVocabulary("</s>");
  
  // Read the words in the file one by one
  long numWordsTrainingFile = 0;
  std::string nextWord = wr.get_next();
  while (!(nextWord.empty())) {
    numWordsTrainingFile++;
    
    // When a word is unknown, add it to the vocabulary
    int i = SearchWordInVocabulary(nextWord);
    if (i == -1) {
      AddWordToVocabulary(nextWord);
    } else {
      // ... otherwise simply increase its count
      m_vocabularyStorage[i].cn++;
    }
    
    // Read next word
    nextWord = wr.get_next();
  }
  
  if (m_usesClassFile) {
    // If we use a class file, the number of word stored in the class map
    // needs to be exactly the same as the number of words in the vocabulary.
    if (m_mapWord2Class.size() < m_vocabularyStorage.size()) {
      printf("Some words in the vocabulary have no class assigned\n");
      return false;
    }
    // Check that every word has a class assigned
    for (size_t i = 0; i < m_vocabularyStorage.size(); i++) {
      const std::string &word = m_vocabularyStorage[i].word;
      if (m_mapWord2Class.find(word) == m_mapWord2Class.end()) {
        printf("%s missing from class file\n", word.c_str());
        return false;
      }
      m_vocabularyStorage[i].classIndex = m_mapWord2Class[word];
    }
    // Sort the words by class
    SortVocabularyByClass();
  } else {
    // Simply sort the words by frequency, making sure that </s> is first
    int countEos = m_vocabularyStorage[0].cn;
    m_vocabularyStorage[0].cn = INT_MAX;
    SortVocabularyByFrequency();
    m_vocabularyStorage[0].cn = countEos;
  }
  
  // Rebuild the the maps of word <-> word index
  m_mapWord2Index.clear();
  m_mapIndex2Word.clear();
  for (int index = 0; index < GetVocabularySize(); index++) {
    string word = m_vocabularyStorage[index].word;
    // Add the word to the hash table word -> index
    m_mapWord2Index[word] = index;
    // Add the word to the hash table index -> word
    m_mapIndex2Word[index] = word;
  }
  
  if (m_debugMode) {
    printf("Vocab size: %d\n", GetVocabularySize());
    printf("Words in train file: %ld\n", numWordsTrainingFile);
  }
  m_numTrainWords = numWordsTrainingFile;
  
  return true;
}


/// <summary>
/// Save a matrix of floats in binary format
/// </summary>
void SaveBinaryMatrix(FILE *fo, int sizeIn, int sizeOut, const vector<double> &vec)
{
  if (sizeIn * sizeOut == 0) {
    return;
  }
  for (int idxOut = 0; idxOut < sizeOut; idxOut++) {
    for (int idxIn = 0; idxIn < sizeIn; idxIn++) {
      float val = (float)(vec[idxIn + idxOut * sizeIn]);
      fwrite(&val, 4, 1, fo);
    }
  }
}


/// <summary>
/// Save a vector of floats in binary format
/// </summary>
void SaveBinaryVector(FILE *fo, long long size, const vector<double> &vec)
{
  for (long long aa = 0; aa < size; aa++) {
    float val = vec[aa];
    fwrite(&val, 4, 1, fo);
  }
}


/// <summary>
/// Once we train the RNN model, it is nice to save it to a text or binary file
/// </summary>
bool RnnLMTraining::SaveRnnModelToFile()
{
  FILE *fo = fopen(m_rnnModelFile.c_str(), "wb");
  if (fo == NULL) {
    printf("Cannot create file %s\n", m_rnnModelFile.c_str());
    return false;
  }
  fprintf(fo, "version: %d\n", m_rnnModelVersion);
  fprintf(fo, "file format: 1\n\n");
  
  fprintf(fo, "training data file: %s\n", m_trainFile.c_str());
  fprintf(fo, "validation data file: %s\n\n", m_validationFile.c_str());
  
  fprintf(fo, "last probability of validation data: %f\n", 0.0);
  fprintf(fo, "number of finished iterations: %d\n", m_iteration);
  
  fprintf(fo, "current position in training data: %ld\n", m_currentPosTrainFile);
  fprintf(fo, "current probability of training data: %f\n", 0.0);
  // dummy used for backward compatibility
  int anti_k = 0;
  fprintf(fo, "save after processing # words: %d\n", anti_k);
  fprintf(fo, "# of training words: %ld\n", m_numTrainWords);
  
  fprintf(fo, "input layer size: %d\n", GetInputSize());
  fprintf(fo, "feature size: %d\n", GetFeatureSize());
  if (!m_featureMatrixUsed) {
    fprintf(fo, "feature matrix used: 0\n");
  } else {
    fprintf(fo, "feature matrix used: 1\n");
  }
  fprintf(fo, "feature gamma: %lf\n", m_featureGammaCoeff);
  fprintf(fo, "hidden layer size: %d\n", GetHiddenSize());
  fprintf(fo, "compression layer size: %d\n", GetCompressSize());
  fprintf(fo, "output layer size: %d\n", GetOutputSize());
  
  fprintf(fo, "direct connections: %d\n", GetNumDirectConnections());
  fprintf(fo, "direct order: %d\n", m_directConnectionOrder);
  
  fprintf(fo, "bptt: %d\n", m_numBpttSteps);
  fprintf(fo, "bptt block: %d\n", m_bpttBlockSize);
  
  fprintf(fo, "vocabulary size: %d\n", GetVocabularySize());
  fprintf(fo, "class size: %d\n", m_numOutputClasses);
  
  fprintf(fo, "old classes: 0\n");
  fprintf(fo, "uses class file: %d\n", m_usesClassFile ? 1 : 0);
  fprintf(fo, "independent sentences mode: %d\n",
          m_areSentencesIndependent ? 1 : 0);
  
  fprintf(fo, "starting learning rate: %f\n", m_initialLearningRate);
  fprintf(fo, "current learning rate: %f\n", m_learningRate);
  fprintf(fo, "learning rate decrease: %d\n", m_doStartReducingLearningRate);
  fprintf(fo, "\n");
  
  // Save the vocabulary, one word per line
  int sizeVocabulary = GetVocabularySize();
  fprintf(fo, "\nVocabulary:\n");
  for (int wordIndex = 0; wordIndex < sizeVocabulary; wordIndex++) {
    int wordCount = m_vocabularyStorage[wordIndex].cn;
    string word = m_vocabularyStorage[wordIndex].word;
    int wordClass = m_vocabularyStorage[wordIndex].classIndex;
    fprintf(fo, "%6d\t%10d\t%s\t%d\n",
            wordIndex, wordCount, word.c_str(), wordClass);
  }
  
  int sizeInput = GetInputSize();
  int sizeFeature = GetFeatureSize();
  int sizeHidden = GetHiddenSize();
  int sizeCompress = GetCompressSize();
  int sizeOutput = GetOutputSize();
  long sizeDirectConnection = GetNumDirectConnections();
  
  if (m_debugMode)
    printf("BINARY Saving %d hidden activations...\n", sizeHidden);
  SaveBinaryVector(fo, sizeHidden, m_state.HiddenLayer);
  // Save the weights U: input -> hidden (i.e., the word embeddings)
  if (m_debugMode)
    printf("BINARY Saving %dx%d input->hidden weights...\n", sizeHidden, sizeInput);
  SaveBinaryMatrix(fo, sizeInput, sizeHidden, m_weights.Input2Hidden);
  // Save the weights W: recurrent hidden -> hidden (i.e., the time-delay)
  if (m_debugMode)
    printf("BINARY Saving %dx%d recurrent hidden->hidden weights...\n", sizeHidden, sizeHidden);
  SaveBinaryMatrix(fo, sizeHidden, sizeHidden, m_weights.Recurrent2Hidden);
  // Save the weights feature -> hidden
  if (m_debugMode)
    printf("BINARY Saving %dx%d feature->hidden weights...\n", sizeHidden, sizeFeature);
  SaveBinaryMatrix(fo, sizeFeature, sizeHidden, m_weights.Features2Hidden);
  // Save the weights G: feature -> output
  if (m_debugMode)
    printf("BINARY Saving %dx%d feature->output weights...\n", sizeOutput, sizeFeature);
  SaveBinaryMatrix(fo, sizeFeature, sizeOutput, m_weights.Features2Output);
  // Save the weights hidden -> compress and compress -> output
  // or simply the weights V: hidden -> output
  if (sizeCompress > 0) {
    if (m_debugMode)
      printf("BINARY Saving %dx%d hidden->compress weights...\n", sizeCompress, sizeHidden);
    SaveBinaryMatrix(fo, sizeHidden, sizeCompress, m_weights.Hidden2Output);
    if (m_debugMode)
      printf("BINARY Saving %dx%d compress->output weights...\n", sizeCompress, sizeOutput);
    SaveBinaryMatrix(fo, sizeCompress, sizeOutput, m_weights.Compress2Output);
  } else {
    if (m_debugMode)
      printf("BINARY Saving %dx%d hidden->output weights...\n", sizeOutput, sizeHidden);
    SaveBinaryMatrix(fo, sizeHidden, sizeOutput, m_weights.Hidden2Output);
  }
  if (sizeDirectConnection > 0) {
    // Save the direct connections
    if (m_debugMode)
      printf("BINARY Saving %ld n-gram connections...\n", sizeDirectConnection);
#ifdef USE_HASHTABLES
#else
    for (long long aa = 0; aa < sizeDirectConnection; aa++) {
      float fl = (float)(m_weights.DirectNGram[aa]);
      fwrite(&fl, 4, 1, fo);
    }
#endif
  }
  // Save the feature matrix
  if (m_featureMatrixUsed) {
    if (m_debugMode)
      printf("BINARY Saving %dx%d feature matrix...\n", sizeFeature, sizeVocabulary);
    SaveBinaryMatrix(fo, sizeFeature, sizeVocabulary, m_featureMatrix);
  }
  fclose(fo);
  return true;
}


/// <summary>
/// Read the classes from a file in the following format:
/// word [TAB] class_index
/// where class index is between 0 and n-1 and there are n classes.
/// </summary>
bool RnnLMTraining::ReadClasses(const string &filename)
{
  FILE *fin = fopen(filename.c_str(), "r");
  if (!fin) {
    printf("Error: unable to open %s\n", filename.c_str());
    return false;
  }
  
  char w[8192];
  int clnum;
  int eos_class = -1;
  int max_class = -1;
  set<string> words;
  while (fscanf(fin, "%s%d", w, &clnum) != EOF) {
    if (!strcmp(w, "<s>")) {
      printf("Error: <s> should not be in vocab\n");
      return false;
    }
    
    m_mapWord2Class[w] = clnum;
    m_classes.insert(clnum);
    words.insert(w);
    
    max_class = (clnum > max_class) ? (clnum) : (max_class);
    eos_class = (string(w) == "</s>") ? (clnum) : (eos_class);
  }
  
  if (eos_class == -1) {
    printf("Error: </s> must be present in the vocabulary\n");
    return false;
  }
  
  if (m_mapWord2Class.size() == 0) {
    printf("Error: Empty class file!\n");
    return false;
  }
  
  // </s> needs to have the highest class index because it needs to come first in the vocabulary...
  for (auto si=words.begin(); si!=words.end(); si++) {
    if (m_mapWord2Class[*si] == eos_class) {
      m_mapWord2Class[*si] = max_class;
    } else {
      if (m_mapWord2Class[*si] == max_class) {
        m_mapWord2Class[*si] = eos_class;
      }
    }
  }
  m_usesClassFile = true;
  return true;
}


/// <summary>
/// Cleans all activations and error vectors, in the input, hidden,
/// compression, feature and output layers, and resets word history
/// </summary>
void RnnLMTraining::ResetAllRnnActivations(RnnState &state) const
{
  // Completely reset the input layer and its gradients
  state.InputLayer.assign(GetInputSize(), 0.0);
  state.InputGradient.assign(GetInputSize(), 0.0);
  
  // Set hidden unit activations to 1.0
  // then the hidden layer to the input (i.e., recurrent connection)
  // Reset the word history
  ResetHiddenRnnStateAndWordHistory(state);
  // N.B.: NextWord assigns 1.0 to the previous state s(t-1)
  // (i.e., to the last units of the input layer
  // in range [sizeVocabulary, sizeInput[)
  // at the beginning of the query/sentence.
  // In the code below, there were using the value of 0.1 instead of 1.0
  // but they would afterwards reset it to 1.0 again... Weird.
  // I just keep 1.0.
  
  // Reset the hidden layer again, this time to 0
  // TODO: could function ResetHiddenRnnStateAndWordHistory do that?
  state.HiddenLayer.assign(GetHiddenSize(), 0.0);
  state.HiddenGradient.assign(GetHiddenSize(), 0.0);
  
  // Reset the compression layer and its gradients
  // TODO: could function ResetHiddenRnnStateAndWordHistory do that?
  state.CompressLayer.assign(GetCompressSize(), 0.0);
  state.CompressGradient.assign(GetCompressSize(), 0.0);
  
  // Reset the output layer and its gradients
  state.OutputLayer.assign(GetOutputSize(), 0.0);
  state.OutputGradient.assign(GetOutputSize(), 0.0);
  
  // Reset the feature vector and its gradients
  state.FeatureLayer.assign(GetFeatureSize(), 0.0);
  state.FeatureGradient.assign(GetFeatureSize(), 0.0);
}


/// <summary>
/// One step of backpropagation of the errors through the RNN
/// (optionally, backpropagation through time, BPTT) and of gradient descent.
/// </summary>
void RnnLMTraining::BackPropagateErrorsThenOneStepGradientDescent(int lastWord, int word)
{
  // No learning step if OOV word
  if (word == -1) {
    return;
  }
  
  // Learning rates, with and without regularization
  double beta = m_regularizationRate * m_learningRate;
  double alpha = m_learningRate;
  // Regularization is done every 10th step
  double coeffSGD = ((m_wordCounter % 10) == 0) ? (1.0 - beta) : 1.0;
  
  // Matrix sizes
  int sizeInput = GetInputSize();
  int sizeFeature = GetFeatureSize();
  int sizeOutput = GetOutputSize();
  int sizeHidden = GetHiddenSize();
  int sizeCompress = GetCompressSize();
  int sizeVocabulary = GetVocabularySize();
  int sizeDirectConnection = GetNumDirectConnections();
  
  // Index at which the words in the current target word class are stored
  int idxWordClass = m_classWords[m_vocabularyStorage[word].classIndex][0];
  // Number of words in that class
  int numWordsInClass =
  static_cast<int>(m_classWords[m_vocabularyStorage[word].classIndex].size());
  
  // Backprop starts with computing the error vectors (gradients w.r.t. loss)
  // for the words in the vocabulary, i.e.,
  // diff(Loss) / diff(Output_k) = diff(log P(word_i)) / diff(Output_k)
  // where P(word_i) = exp (Output(word_i)) / sum_j exp(Output(word_j))
  // and word_i is the target word and word_j is any word of the vocabulary.
  // Hence log P(word_i) = Output(word_i) - log sum_j exp(Output(word_j)
  // The gradient vector becomes, for any output k:
  // diff(log P(word_i)) / diff(Output_k) = diff(Output(word_i)) / diff(Output_k)
  //   - diff(log sum_j exp(Output(word_j))) / diff(Output_k)
  // 1) Backprop on words within the target class
  for (int c = 0; c < numWordsInClass; c++) {
    int a = m_classWords[m_vocabularyStorage[word].classIndex][c];
    m_state.OutputGradient[a] = (0 - m_state.OutputLayer[a]);
  }
  m_state.OutputGradient[word] = (1 - m_state.OutputLayer[word]);
  
  // 2) Backprop on all classes
  for (int a = sizeVocabulary; a < sizeOutput; a++) {
    m_state.OutputGradient[a] = (0 - m_state.OutputLayer[a]);
  }
  int wordClassIdx = m_vocabularyStorage[word].classIndex + sizeVocabulary;
  m_state.OutputGradient[wordClassIdx] = (1 - m_state.OutputLayer[wordClassIdx]);
  
  // Reset gradients on hidden layers
  m_state.HiddenGradient.assign(sizeHidden, 0);
  m_state.CompressGradient.assign(sizeCompress, 0);
  
  // learn direct connections between words
  if (sizeDirectConnection > 0) {
    if (word != -1) {
#ifdef USE_HASHTABLES
      for (int c = 0; c < numWordsInClass; c++) {
        int wordInClass = m_classWords[m_vocabularyStorage[word].classIndex][c];
        WordTripleKey key3(wordInClass, m_state.WordHistory[0], m_state.WordHistory[1]);
        if (key3.isValid()) {
          if (m_weights.DirectTriGram.find(key3) == m_weights.DirectTriGram.end()) {
            m_weights.DirectTriGram.insert(pair<WordTripleKey, double>(key3, 0));
          }
          m_weights.DirectTriGram[key3] +=
          alpha * m_state.OutputGradient[wordInClass] - beta * m_weights.DirectTriGram[key3];
        }
        WordPairKey key2(wordInClass, m_state.WordHistory[0]);
        if (key2.isValid()) {
          if (m_weights.DirectBiGram.find(key2) == m_weights.DirectBiGram.end()) {
            m_weights.DirectBiGram.insert(pair<WordPairKey, double>(key2, 0));
          }
          m_weights.DirectBiGram[key2] +=
          alpha * m_state.OutputGradient[wordInClass] - beta * m_weights.DirectBiGram[key2];
        }
      }
#else
      unsigned long long hash[c_maxNGramOrder];
      for (int a = 0; a < m_directConnectionOrder; a++) {
        hash[a] = 0;
      }
      for (int a = 0; a < m_directConnectionOrder; a++) {
        int b = 0;
        if ((a > 0) && (m_state.WordHistory[a-1] == -1)) {
          break;
        }
        hash[a] = c_Primes[0]*c_Primes[1]*(unsigned long long)(m_vocabularyStorage[word].classIndex+1);
        for (b=1; b<=a; b++) {
          hash[a]+=c_Primes[(a*c_Primes[b]+b)%c_PrimesSize]*(unsigned long long)(m_state.WordHistory[b-1]+1);
        }
        hash[a] = (hash[a]%(sizeDirectConnection/2))+(sizeDirectConnection)/2;
      }
      for (int c = 0; c<numWordsInClass; c++) {
        int a = m_classWords[m_vocabularyStorage[word].classIndex][c];
        for (int b = 0; b < m_directConnectionOrder; b++) {
          if (hash[b]) {
            m_weights.DirectNGram[hash[b]] +=
            alpha * m_state.OutputGradient[a] - m_weights.DirectNGram[hash[b]]*beta;
            hash[b]++;
            hash[b] = hash[b]%sizeDirectConnection;
          } else {
            break;
          }
        }
      }
#endif
    }
  }
  //
  // learn direct connections to classes
  if (sizeDirectConnection > 0) {
#ifdef USE_HASHTABLES
    for (int c = sizeVocabulary; c < sizeOutput; c++) {
      WordTripleKey key3(c, m_state.WordHistory[0], m_state.WordHistory[1]);
      if (key3.isValid()) {
        if (m_weights.DirectTriGram.find(key3) == m_weights.DirectTriGram.end()) {
          m_weights.DirectTriGram.insert(pair<WordTripleKey, double>(key3, 0));
        }
        m_weights.DirectTriGram[key3] +=
        alpha * m_state.OutputGradient[c] - beta * m_weights.DirectTriGram[key3];
      }
      WordPairKey key2(c, m_state.WordHistory[0]);
      if (key2.isValid()) {
        if (m_weights.DirectBiGram.find(key2) == m_weights.DirectBiGram.end()) {
          m_weights.DirectBiGram.insert(pair<WordPairKey, double>(key2, 0));
        }
        m_weights.DirectBiGram[key2] +=
        alpha * m_state.OutputGradient[c] - beta * m_weights.DirectBiGram[key2];
      }
    }
#else
    unsigned long long hash[c_maxNGramOrder] = {0};
    for (int a = 0; a < m_directConnectionOrder; a++) {
      int b = 0;
      if (a>0) if (m_state.WordHistory[a-1] == -1) break;
      hash[a] = c_Primes[0]*c_Primes[1];
      for (b=1; b<=a; b++) {
        hash[a] += c_Primes[(a*c_Primes[b]+b)%c_PrimesSize]*(unsigned long long)(m_state.WordHistory[b-1]+1);
      }
      hash[a] = hash[a]%(sizeDirectConnection/2);
    }
    for (int a = sizeVocabulary; a < sizeOutput; a++) {
      for (int b = 0; b < m_directConnectionOrder; b++) {
        if (hash[b]) {
          m_weights.DirectNGram[hash[b]] +=
          alpha * m_state.OutputGradient[a] - m_weights.DirectNGram[hash[b]]*beta;
          hash[b]++;
        } else {
          break;
        }
      }
    }
#endif
  }
  
  if (sizeCompress > 0) {
    // Back-propagate gradients coming from loss on words in target class
    // w.r.t. the compression layer
    GradientMatrixXvectorBlas(m_state.CompressGradient,
                              m_state.OutputGradient,
                              m_weights.Compress2Output,
                              sizeCompress,
                              idxWordClass,
                              idxWordClass + numWordsInClass);
    
    // Back-propagate gradients coming from loss on words in target class
    // w.r.t. the weights V between the compression layer and the output word layer
    // V[[classIdx, classIdx+numWordsClass] x [1, sizeHidden]]
    //   <- (1-beta) * V[[classIdx, classIdx+numWordsClass] x [1, sizeHidden]]
    //      + alpha * dOut[[classIdx, classIdx+numWordsClass], 1] * c(t)[1, [1, sizeHidden]]
    MultiplyMatrixXmatrixBlas(m_state.OutputGradient,
                              m_state.CompressLayer,
                              m_weights.Compress2Output,
                              alpha,
                              coeffSGD,
                              sizeOutput,
                              1,
                              sizeCompress,
                              idxWordClass,
                              idxWordClass + numWordsInClass);
    
    // Back-propagate gradients coming from loss on word classes
    // w.r.t. the compression layer
    GradientMatrixXvectorBlas(m_state.CompressGradient,
                              m_state.OutputGradient,
                              m_weights.Compress2Output,
                              sizeCompress,
                              sizeVocabulary,
                              sizeOutput);
    
    // Back-propagate gradients coming from loss on word classes
    // w.r.t. the weights V between the compression layer and the output word layer
    // V[[sizeVocabulary, sizeOutput] x [1, sizeHidden]]
    //   <- (1-beta) * V[[sizeVocabulary, sizeOutput] x [1, sizeHidden]]
    //      + alpha * dOut[[sizeVocabulary, sizeOutput], 1] * c(t)[1, [1, sizeHidden]]
    MultiplyMatrixXmatrixBlas(m_state.OutputGradient,
                              m_state.CompressLayer,
                              m_weights.Compress2Output,
                              alpha,
                              coeffSGD,
                              sizeOutput,
                              1,
                              sizeCompress,
                              sizeVocabulary,
                              sizeOutput);
    
    // Back-propagate gradients coming from loss on compression layer
    // w.r.t. the hidden layer
    GradientMatrixXvectorBlas(m_state.HiddenGradient,
                              m_state.CompressGradient,
                              m_weights.Hidden2Output,
                              sizeHidden,
                              0,
                              sizeCompress);
    
    // Back-propagate gradients coming from loss on compression layer
    // w.r.t. the weights V between the hidden layer and the compression layer
    // V[[1, sizeHidden] x [1, sizeHidden]]
    //   <- (1-beta) * V[[1, sizeHidden] x [1, sizeHidden]]
    //      + alpha * dc(t)[[1, sizeHidden], 1] * h(t)[1, [1, sizeHidden]]
    MultiplyMatrixXmatrixBlas(m_state.CompressGradient,
                              m_state.HiddenLayer,
                              m_weights.Hidden2Output,
                              alpha,
                              1.0,
                              sizeHidden,
                              1,
                              sizeCompress,
                              0,
                              sizeHidden);
  } else {
    // Back-propagate gradients coming from loss on words in target class
    // w.r.t. the hidden layer
    GradientMatrixXvectorBlas(m_state.HiddenGradient,
                              m_state.OutputGradient,
                              m_weights.Hidden2Output,
                              sizeHidden,
                              idxWordClass,
                              idxWordClass + numWordsInClass);
    
    // Back-propagate gradients coming from loss on words in target class
    // w.r.t. the weights V between the hidden layer and the output word layer
    // V[[classIdx, classIdx+numWordsClass] x [1, sizeHidden]]
    //   <- (1-beta) * V[[classIdx, classIdx+numWordsClass] x [1, sizeHidden]]
    //      + alpha * dOut[[classIdx, classIdx+numWordsClass], 1] * h(t)[1, [1, sizeHidden]]
    MultiplyMatrixXmatrixBlas(m_state.OutputGradient,
                              m_state.HiddenLayer,
                              m_weights.Hidden2Output,
                              alpha,
                              coeffSGD,
                              sizeOutput,
                              1,
                              sizeHidden,
                              idxWordClass,
                              idxWordClass + numWordsInClass);
    
    // Back-propagate gradients coming from loss on word classes
    // w.r.t. the hidden layer
    GradientMatrixXvectorBlas(m_state.HiddenGradient,
                              m_state.OutputGradient,
                              m_weights.Hidden2Output,
                              sizeHidden,
                              sizeVocabulary,
                              sizeOutput);
    
    // Back-propagate gradients coming from loss on word classes
    // w.r.t. the weights V between the hidden layer and the output word layer
    // V[[sizeVocabulary, sizeOutput] x [1, sizeHidden]]
    //   <- (1-beta) * V[[sizeVocabulary, sizeOutput] x [1, sizeHidden]]
    //      + alpha * dOut[[sizeVocabulary, sizeOutput], 1] * h(t)[1, [1, sizeHidden]]
    MultiplyMatrixXmatrixBlas(m_state.OutputGradient,
                              m_state.HiddenLayer,
                              m_weights.Hidden2Output,
                              alpha,
                              coeffSGD,
                              sizeOutput,
                              1,
                              sizeHidden,
                              sizeVocabulary,
                              sizeOutput);
  }
  
  if (sizeFeature > 0) {
    // Back-propagate gradients coming from loss on words in target class
    // w.r.t. the weights V between the hidden layer and the output word layer
    // G[[classIdx, classIdx+numWordsClass] x [1, sizeFeature]]
    //   <- G[[classIdx, classIdx+numWordsClass] x [1, sizeFeature]]
    //      + alpha * dOut[[classIdx, classIdx+numWordsClass], 1] * f(t)[1, [1, sizeFeature]]
    MultiplyMatrixXmatrixBlas(m_state.OutputGradient,
                              m_state.FeatureLayer,
                              m_weights.Features2Output,
                              alpha,
                              1.0,
                              sizeOutput,
                              1,
                              sizeFeature,
                              idxWordClass,
                              idxWordClass + numWordsInClass);
    
    // Back-propagate gradients coming from loss on word classes
    // w.r.t. the direct weights G between the feature layer and the output word layer
    // G[[sizeVocabulary, sizeOutput] x [1, sizeFeature]]
    //   <- G[[sizeVocabulary, sizeOutput] x [1, sizeFeature]]
    //      + alpha * dOut[[sizeVocabulary, sizeOutput], 1] * f(t)[1, [1, sizeFeature]]
    MultiplyMatrixXmatrixBlas(m_state.OutputGradient,
                              m_state.FeatureLayer,
                              m_weights.Features2Output,
                              alpha,
                              1.0,
                              sizeOutput,
                              1,
                              sizeFeature,
                              sizeVocabulary,
                              sizeOutput);
  }
  
  if (m_numBpttSteps <= 1) {
    // bptt==1 -> normal BP
    for (int a = 0; a < sizeHidden; a++) {
      double dLdSa = m_state.HiddenLayer[a];
      m_state.HiddenGradient[a] = m_state.HiddenGradient[a] * dLdSa * (1 - dLdSa);
      // error derivation at layer 1
    }
    
    // weight update hidden(t) -> input(t)
    int a = lastWord;
    if (a != -1) {
      for (int b = 0; b < sizeHidden; b++) {
        int node = a + b * sizeInput;
        m_weights.Input2Hidden[node] =
        alpha * m_state.HiddenGradient[b] * m_state.InputLayer[a]
        + coeffSGD * m_weights.Input2Hidden[node];
      }
    }
    
    // weight update hidden(t) -> hidden(t-1)
    MultiplyMatrixXmatrixBlas(m_state.HiddenGradient,
                              m_state.RecurrentLayer,
                              m_weights.Recurrent2Hidden,
                              alpha,
                              coeffSGD,
                              sizeHidden,
                              1,
                              sizeHidden,
                              0,
                              sizeHidden);
    
    // weight update hidden(t) -> feature(t)
    MultiplyMatrixXmatrixBlas(m_state.HiddenGradient,
                              m_state.FeatureLayer,
                              m_weights.Features2Hidden,
                              alpha,
                              coeffSGD,
                              sizeHidden,
                              1,
                              sizeFeature,
                              0,
                              sizeHidden);
  } else {
    // BPTT
    for (int b = 0; b < sizeHidden; b++) {
      m_bpttVectors.HiddenLayer[b] = m_state.HiddenLayer[b];
    }
    for (int b = 0; b < sizeHidden; b++) {
      m_bpttVectors.HiddenGradient[b] = m_state.HiddenGradient[b];
    }
    for (int b = 0; b < sizeFeature; b++) {
      m_bpttVectors.FeatureLayer[b] = m_state.FeatureLayer[b];
    }
    
    if (((m_wordCounter % m_bpttBlockSize) == 0) || (m_areSentencesIndependent && (word == 0))) {
      for (int step = 0; step<m_numBpttSteps+m_bpttBlockSize-2; step++) {
        for (int a = 0; a < sizeHidden; a++) {
          double dLdSa = m_state.HiddenLayer[a];
          m_state.HiddenGradient[a] = m_state.HiddenGradient[a] * dLdSa * (1 - dLdSa);
          // error derivation at layer 1
        }
        
        if (sizeFeature > 0) {
          // weight update hidden(t) -> feature(t)
          MultiplyMatrixXmatrixBlas(m_state.HiddenGradient,
                                    m_state.FeatureLayer,
                                    m_bpttVectors.WeightsFeature2Hidden,
                                    alpha,
                                    1.0,
                                    sizeHidden,
                                    1,
                                    sizeFeature,
                                    0,
                                    sizeHidden);
        }
        
        // weight update hidden -> input
        int a = m_bpttVectors.History[step];
        if (a != -1) {
          for (int b = 0; b < sizeHidden; b++)
          {
            m_bpttVectors.WeightsInput2Hidden[a + b * sizeInput] +=
            alpha * m_state.HiddenGradient[b];
          }
        }
        
        // weight update hidden -> recurrent
        m_state.HiddenGradient.assign(sizeHidden, 0);
        GradientMatrixXvectorBlas(m_state.RecurrentGradient,
                                  m_state.HiddenGradient,
                                  m_weights.Recurrent2Hidden,
                                  sizeHidden,
                                  0,
                                  sizeHidden);
        
        MultiplyMatrixXmatrixBlas(m_state.HiddenGradient,
                                  m_state.RecurrentLayer,
                                  m_bpttVectors.WeightsRecurrent2Hidden,
                                  alpha,
                                  1.0,
                                  sizeHidden,
                                  1,
                                  sizeHidden,
                                  0,
                                  sizeHidden);
        
        for (int a = 0; a < sizeHidden; a++) {
          // propagate error from time T-n to T-n-1
          m_state.HiddenGradient[a] =
          m_state.RecurrentGradient[a] +
          m_bpttVectors.HiddenGradient[(step+1) * sizeHidden+a];
        }
        
        if (step < m_numBpttSteps + m_bpttBlockSize - 3) {
          for (int a = 0; a < sizeHidden; a++)
          {
            m_state.HiddenLayer[a] =
            m_bpttVectors.HiddenLayer[(step+1) * sizeHidden+a];
            m_state.RecurrentLayer[a] =
            m_bpttVectors.HiddenLayer[(step+2) * sizeHidden+a];
          }
        }
      }
      
      for (int a = 0; a < (m_numBpttSteps+m_bpttBlockSize) * sizeHidden; a++) {
        m_bpttVectors.HiddenGradient[a] = 0;
      }
      
      for (int b = 0; b < sizeHidden; b++) {
        m_state.HiddenLayer[b] = m_bpttVectors.HiddenLayer[b];
        // restore hidden layer after bptt
      }
      
      //
      AddMatrixToMatrixBlas(m_bpttVectors.WeightsRecurrent2Hidden,
                            m_weights.Recurrent2Hidden,
                            1.0,
                            coeffSGD,
                            sizeHidden,
                            sizeHidden);
      m_bpttVectors.WeightsRecurrent2Hidden.assign(sizeHidden * sizeHidden, 0);
      
      if (sizeFeature > 0) {
        AddMatrixToMatrixBlas(m_bpttVectors.WeightsFeature2Hidden,
                              m_weights.Features2Hidden,
                              1.0,
                              coeffSGD,
                              sizeHidden,
                              sizeFeature);
        m_bpttVectors.WeightsFeature2Hidden.assign(sizeHidden * sizeFeature, 0);
      }
      
      for (int step = 0; step < m_numBpttSteps + m_bpttBlockSize - 2; step++) {
        int wordAtStep = m_bpttVectors.History[step];
        if (wordAtStep != -1) {
          for (int b = 0; b < sizeHidden; b++)
          {
            int node = wordAtStep + b * sizeInput;
            m_weights.Input2Hidden[node] =
            m_bpttVectors.WeightsInput2Hidden[node]
            + coeffSGD * m_weights.Input2Hidden[node];
            m_bpttVectors.WeightsInput2Hidden[node] = 0;
          }
        }
      }
    }
  }
}


/// <summary>
/// Train a Recurrent Neural Network model on a test file
/// </summary>
bool RnnLMTraining::TrainRnnModel()
{
  // Do we use an external file with feature vectors for each
  // consecutive word in the test set?
  // Only if feature matrix (LDA/LSA topic model or Word2Vec) was not set
  // and there is a feature file
  bool isFeatureFileUsed =
  ((!m_featureMatrixUsed) && !m_featureFile.empty());
  FILE *featureFileId = NULL;
  int sizeFeature = GetFeatureSize();
  
  // Reset the log-likelihood of the last iteration to ginourmous value
  double lastValidLogProbability = -1E37;
  // Word counter, saved at the end of last training session
  m_wordCounter = (int)m_currentPosTrainFile;
  // Keep track of the initial learning rate
  m_initialLearningRate = m_learningRate;
  
  // Sanity check
  if (m_numOutputClasses > GetVocabularySize()) {
    printf("WARNING: number of classes exceeds vocabulary size!\n");
  }
  
  bool loopEpochs = true;
  while (loopEpochs) {
    // Reset the log-likelihood of the current iteration
    double trainLogProbability = 0.0;
    
    // Create a word reader on the training file
    WordReader wordReaderTrain(m_trainFile);
    printf("Starting training using file %s\n", m_trainFile.c_str());
    // Verbose
    printf("Iter: %3d\tAlpha: %f\n", m_iteration, m_learningRate);
    fflush(stdout);
    
    // Reset everything, including word history
    ResetAllRnnActivations(m_state);
    
    // Ugly way to open the feature vector file
    if (isFeatureFileUsed) {
      if (m_featureFile.empty()) {
        printf("Feature file for the test data is needed to evaluate this model (use -features <FILE>)\n");
        return 0;
      }
      featureFileId = fopen(m_featureFile.c_str(), "rb");
      int dummySizeFeature;
      fread(&dummySizeFeature, sizeof(dummySizeFeature), 1, featureFileId);
    }
    
    // Last word set to end of sentence
    int lastWord = 0;
    // Current word
    int word = 0;
    
    // Skip the first m_wordCounter words
    // (if the training was interrupted in the middle of an epoch)
    if (m_wordCounter > 0) {
      for (int a = 0; a < m_wordCounter; a++) {
        word = ReadWordIndexFromFile(wordReaderTrain);
      }
    }
    
    // Start an iteration
    clock_t start = clock();
    bool loopTrain = true;
    while (loopTrain) {
      // Verbose
      if ((m_wordCounter % 10000 == 0) && (m_wordCounter > 0)) {
        clock_t now = clock();
        double entropy = -trainLogProbability/log10((double)2) / m_wordCounter;
        if (m_numTrainWords > 0) {
          printf("Iter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %.2f%%   Words/sec: %.1f\n", m_iteration, m_learningRate, entropy, m_wordCounter/(double)m_numTrainWords*100, m_wordCounter/((double)(now-start)/1000.0));
        } else {
          printf("Iter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %ldK\n", m_iteration, m_learningRate, entropy, m_wordCounter/1000);
        }
      }
      
      // Read next word
      word = ReadWordIndexFromFile(wordReaderTrain);
      loopTrain = (word >= -1);
      if (loopTrain) {
        // Use the pre-computed feature file?
        if (isFeatureFileUsed) {
          LoadFeatureVectorAtCurrentWord(featureFileId, m_state);
        }
        // Use the topic-model features coming from a word embedding matrix?
        if (m_featureMatrixUsed) {
          UpdateFeatureVectorUsingTopicModel(lastWord, m_state);
        }
        
        // Run one step of the RNN
        ForwardPropagateOneStep(lastWord, word, m_state);
        
        // For perplexity, we do not to count OOV words...
        if (word >= 0) {
          // Compute the log-probability of the current word
          int outputNodeClass =
          m_vocabularyStorage[word].classIndex + GetVocabularySize();
          double condProbaClass = m_state.OutputLayer[outputNodeClass];
          double condProbaWordGivenClass =  m_state.OutputLayer[word];
          trainLogProbability +=
          log10(condProbaClass * condProbaWordGivenClass);
          m_wordCounter++;
        }
        
        // Safety check (that log-likelihood does not diverge)
        if (trainLogProbability != trainLogProbability) {
          // || (isinf(trainLogProbability)
          printf("\nNumerical error infinite log-likelihood\n");
          return false;
        }
        
        // Shift memory needed for BPTT to next time step
        if (m_numBpttSteps > 0) {
          // shift memory needed for bptt to next time step
          for (int a = m_numBpttSteps+m_bpttBlockSize-1; a > 0; a--)
          {
            m_bpttVectors.History[a] = m_bpttVectors.History[a-1];
          }
          m_bpttVectors.History[0] = lastWord;
          
          int sizeHidden = GetHiddenSize();
          for (int a = m_numBpttSteps+m_bpttBlockSize-1; a > 0; a--)
          {
            for (int b = 0; b < sizeHidden; b++)
            {
              m_bpttVectors.HiddenLayer[a * sizeHidden+b] = m_bpttVectors.HiddenLayer[(a-1) * sizeHidden+b];
              m_bpttVectors.HiddenGradient[a * sizeHidden+b] = m_bpttVectors.HiddenGradient[(a-1) * sizeHidden+b];
            }
          }
          
          for (int a = m_numBpttSteps+m_bpttBlockSize-1; a>0; a--)
          {
            for (int b = 0; b < sizeFeature; b++)
            {
              m_bpttVectors.FeatureLayer[a * sizeFeature+b] = m_bpttVectors.FeatureLayer[(a-1) * sizeFeature+b];
            }
          }
        }
        
        // Back-propagate the error and run one step of
        // stochastic gradient descent (SGD) using optional
        // back-propagation through time (BPTT)
        BackPropagateErrorsThenOneStepGradientDescent(lastWord, word);
        
        // Store the current state s(t) at the end of the input layer vector
        // so that it can be used as s(t-1) at the next step
        ForwardPropagateRecurrentConnectionOnly(m_state);
        
        // Rotate the word history by one
        ForwardPropagateWordHistory(m_state, lastWord, word);
        
        // Did we reach the end of the sentence?
        // If so, we need to reset the state of the neural net
        if (m_areSentencesIndependent && (word == 0)) {
          ResetHiddenRnnStateAndWordHistory(m_state);
        }
      }
    }
    
    // Close the feature file
    if (isFeatureFileUsed) {
      fclose(featureFileId);
    }
    
    // Verbose
    clock_t now = clock();
    double trainEntropy = -trainLogProbability/log10((double)2) / m_wordCounter;
    double trainPerplexity =
    ExponentiateBase10(-trainLogProbability / (double)m_wordCounter);
    printf("Iter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    PPX: %.4f   Words/sec: %.1f\n", m_iteration, m_learningRate, trainEntropy, trainPerplexity, m_wordCounter/((double)(now-start)/1000.0));
    
    // Validation
    int validWordCounter = 0;
    vector<double> sentenceScores;
    double validLogProbability =
    TestRnnModel(m_validationFile,
                 m_featureValidationFile,
                 validWordCounter,
                 sentenceScores);
    printf("Training iteration: %d\n", m_iteration);
    printf("VALID log probability: %f\n", validLogProbability);
    double validPerplexity =
    (validWordCounter == 0) ? 0 :
    ExponentiateBase10(-validLogProbability / (double)validWordCounter);
    printf("VALID PPL net less OOVs: %f\n", validPerplexity);
    double validEntropy =
    (validWordCounter == 0) ? 0 :
    -validLogProbability / log10((double)2) / validWordCounter;
    printf("VALID entropy: %.4f\n", validEntropy);
    
    // Reset the position in the training file
    m_wordCounter = 0;
    m_currentPosTrainFile = 0;
    trainLogProbability = 0;
    
    if (validLogProbability < lastValidLogProbability) {
      // Restore the weights and the state from the backup
      m_weights = m_weightsBackup;
      m_state = m_stateBackup;
      printf("Restored the weights from previous iteration\n");
    } else {
      // Backup the weights and the state
      m_weightsBackup = m_weights;
      m_stateBackup = m_state;
      printf("Save this model\n");
    }
    
    // Shall we start reducing the learning rate?
    if (validLogProbability * m_minLogProbaImprovement < lastValidLogProbability) {
      if (!m_doStartReducingLearningRate) {
        m_doStartReducingLearningRate = true;
      } else {
        SaveRnnModelToFile();
        // Let's also save the word embeddings
        SaveWordEmbeddings(m_rnnModelFile + ".word_embeddings.txt");
        loopEpochs = false;
        break;
      }
    }
    
    if (loopEpochs) {
      if (m_doStartReducingLearningRate) {
        m_learningRate /= 2;
      }
      lastValidLogProbability = validLogProbability;
      validLogProbability = 0;
      m_iteration++;
      SaveRnnModelToFile();
      // Let's also save the word embeddings
      SaveWordEmbeddings(m_rnnModelFile + ".word_embeddings.txt");
      printf("Saved the model\n");
    }
  }
  
  return true;
}


/// <summary>
/// Test a Recurrent Neural Network model on a test file
/// </summary>
double RnnLMTraining::TestRnnModel(const string &testFile,
                                   const string &featureFile,
                                   int &wordCounter,
                                   vector<double> &sentenceScores)
{
  // Do we use an external file with feature vectors for each
  // consecutive word in the test set?
  // Only if feature matrix (LDA/LSA topic model or Word2Vec) was not set
  // and there is a feature file
  bool isFeatureFileUsed =
  ((!m_featureMatrixUsed) && !featureFile.empty());
  FILE *featureFileId = NULL;
  int sizeFeature = GetFeatureSize();
  
  // This function does what ResetHiddenRnnStateAndWordHistory does
  // and also resets the features, inputs, outputs and compression layer
  ResetAllRnnActivations(m_state);
  
  // Create a word reader on the test file
  WordReader wordReaderTest(testFile);
  
  // Ugly way to open the feature vector file
  if (isFeatureFileUsed) {
    if (featureFile.empty()) {
      printf("Feature file for the test data is needed to evaluate this model (use -features <FILE>)\n");
      return 0;
    }
    featureFileId = fopen(featureFile.c_str(), "rb");
    int a;
    fread(&a, sizeof(a), 1, featureFileId);
    if (a != sizeFeature) {
      printf("Mismatch between feature vector size in model file and feature file (model uses %d features, in %s found %d features)\n", sizeFeature, m_featureFile.c_str(), a);
      return 0;
    }
  }
  
  if (m_debugMode) {
    cout << "Index   P(NET)          Word\n";
    cout << "----------------------------------\n";
  }
  
  // Last word set to end of sentence
  int lastWord = 0;
  // Reset the log-likelihood
  double logProbability = 0.0;
  double sentenceLogProbability = 0.0;
  // Reset the word counter
  wordCounter = 0;
  // Reset the sentence scores
  sentenceScores.clear();
  
  // Since we just set s(1)=0, this will set the state s(t-1) to 0 as well...
  ForwardPropagateRecurrentConnectionOnly(m_state);
  // OK...
  ResetWordHistory(m_state);
  if (m_areSentencesIndependent) {
    // OK, let's reset the hidden states again to 1.0
    // and copy that value to the s(t-1) in the inputs
    // How many times are we going to reset that RNN?
    ResetHiddenRnnStateAndWordHistory(m_state);
  }
  
  // Iterate over the test file
  bool loopTest = true;
  while (loopTest) {
    // Get the index of the next word (or -1 if OOV or -2 if end of file)
    int word = ReadWordIndexFromFile(wordReaderTest);
    loopTest = (word >= -1);
    
    if (loopTest) {
      // Use the pre-computed feature file?
      if (isFeatureFileUsed) {
        LoadFeatureVectorAtCurrentWord(featureFileId, m_state);
      }
      // Use the topic-model features coming from a word embedding matrix?
      if (m_featureMatrixUsed) {
        UpdateFeatureVectorUsingTopicModel(lastWord, m_state);
      }
      
      // Run one step of the RNN
      ForwardPropagateOneStep(lastWord, word, m_state);
      
      // For perplexity, we do not count OOV words...
      if (word >= 0) {
        // Compute the log-probability of the current word
        int outputNodeClass =
        m_vocabularyStorage[word].classIndex + GetVocabularySize();
        double condProbaClass = m_state.OutputLayer[outputNodeClass];
        double condProbaWordGivenClass =  m_state.OutputLayer[word];
        logProbability +=
        log10(condProbaClass * condProbaWordGivenClass);
        sentenceLogProbability +=
        log10(condProbaClass * condProbaWordGivenClass);
        wordCounter++;
        if (m_debugMode) {
          cout << word << "\t"
          << condProbaClass * condProbaWordGivenClass << "\t"
          << m_vocabularyStorage[word].word << endl;
        }
      } else {
        if (m_debugMode) {
          // Out-of-vocabulary words have probability 0 and index -1
          cout << "-1\t0\tOOV\n";
        }
      }
      
      // Store the current state s(t) at the end of the input layer vector
      // so that it can be used as s(t-1) at the next step
      ForwardPropagateRecurrentConnectionOnly(m_state);
      
      // Rotate the word history by one
      ForwardPropagateWordHistory(m_state, lastWord, word);
      
      // Did we reach the end of the sentence?
      // If so, we need to reset the state of the neural net
      // and to save the current sentence score
      if (m_areSentencesIndependent && (word == 0)) {
        ResetHiddenRnnStateAndWordHistory(m_state);
        cout << sentenceLogProbability << endl;
        sentenceScores.push_back(sentenceLogProbability);
        sentenceLogProbability = 0.0;
      }
    }
  }
  
  if (isFeatureFileUsed) {
    fclose(featureFileId);
  }
  
  // Return the total logProbability
  printf("Log probability: %.2f, number of words: %d (%lu sentences)\n",
         logProbability, wordCounter, sentenceScores.size());
  double perplexity =
  (wordCounter == 0) ? 0 : ExponentiateBase10(-logProbability / (double)wordCounter);
  cout << "PPL net (perplexity without OOV): " << perplexity << endl;
  return logProbability;
}


/// <summary>
/// Load a file containing the classification labels
/// </summary>
void RnnLMTraining::LoadCorrectSentenceLabels(const std::string &labelFile)
{
  m_correctSentenceLabels.clear();
  ifstream file(labelFile, ifstream::in);
  int label = 0;
  while (file >> label) {
    m_correctSentenceLabels.push_back(label);
  }
  cout << "Loaded correct labels for " << m_correctSentenceLabels.size()
       << " validation/test sentences\n";
  file.close();
}


/// <summary>
/// Compute the accuracy of selecting the top candidate (based on score)
/// among n-best lists
/// </summary>
double RnnLMTraining::AccuracyNBestList(std::vector<double> scores,
                                        std::vector<int> &correctLabels) const
{
  // Size of the n-best list
  int numSentences = (int)(correctLabels.size());
  assert(scores.size() % numSentences == 0);
  int n = (int)(scores.size()) / numSentences;
  
  // Loop over sentences
  int numAccurate = 0;
  for (int k = 0; k < numSentences; k++) {
    // Compute maximum inside the n-best list for the current sentence
    int j = k * n;
    double bestScore = scores[j];
    double bestIdx = 0;
    for (int idx = 1; idx < n; idx++) {
      if (scores[j+idx] > bestScore) {
        bestScore = scores[j+idx];
        bestIdx = idx;
      }
    }
    
    // Compute accuracy
    numAccurate += (bestIdx == correctLabels[k]);
  }
  
  // Return accuracy
  return ((double)numAccurate / numSentences);
}


/// <summary>
/// Read the feature vector for the current word
/// in the train/test/valid file and update the feature vector
/// in the state
/// TODO: convert to ifstream
/// </summary>
bool RnnLMTraining::LoadFeatureVectorAtCurrentWord(FILE *f,
                                                   RnnState &state)
{
  int sizeFeature = GetFeatureSize();
  for (int a = 0; a < sizeFeature; a++) {
    float fl;
    if (fread(&fl, sizeof(fl), 1, f) != 1) {
      // reached end of file
      return false;
    }
    m_state.FeatureLayer[a] = fl;
  }
  return true;
}


/// <summary>
/// Simply write the word projections/embeddings to a text file.
/// </summary>
void RnnLMTraining::SaveWordEmbeddings(const string &filename)
{
  FILE *fid = fopen(filename.c_str(), "wb");
  
  fprintf(fid, "%d %d\n", GetVocabularySize(), GetHiddenSize());
  
  for (int a = 0; a < GetVocabularySize(); a++) {
    fprintf(fid, "%s ", m_vocabularyStorage[a].word.c_str());
    for (int b = 0; b < GetHiddenSize(); b++) {
      fprintf(fid, "%lf ", m_weights.Input2Hidden[a + b * GetInputSize()]);
    }
    fprintf(fid, "\n");
  }
  
  fclose(fid);
}


/// <summary>
/// Matrix-vector multiplication routine, somewhat accelerated using loop
/// unrolling over 8 registers. Computes x <- x + A' * y,
/// i.e., the "inverse" operation to y = A * x (adding the result to x)
/// where A is of size N x M, x is of length M and y is of length N.
/// The operation can done on a contiguous subset of indices
/// j in [idxYFrom, idxYTo[ of vector y.
/// </summary>
void RnnLMTraining::GradientMatrixXvectorBlas(vector<double> &vectorX,
                                              vector<double> &vectorY,
                                              vector<double> &matrixA,
                                              int widthMatrix,
                                              int idxYFrom,
                                              int idxYTo) const
{
  double *vecX = &vectorX[0];
  int idxAFrom = idxYFrom * widthMatrix;
  double *matA = &matrixA[idxAFrom];
  int heightMatrix = idxYTo - idxYFrom;
  double *vecY = &vectorY[idxYFrom];
  cblas_dgemv(CblasRowMajor, CblasTrans,
              heightMatrix, widthMatrix, 1.0, matA, widthMatrix,
              vecY, 1,
              1.0, vecX, 1);
  // The point of gradient cutoff is to avoid too large values
  // being sent down the RNN, making the learning unstable
  if (m_gradientCutoff > 0) {
    for (int a = 0; a < widthMatrix; a++) {
      if (vectorX[a] > m_gradientCutoff) {
        vectorX[a] = m_gradientCutoff;
      }
      if (vectorX[a] < -m_gradientCutoff) {
        vectorX[a] = -m_gradientCutoff;
      }
    }
  }
}


/// <summary>
/// Matrix-matrix multiplication routine using BLAS.
/// Computes C <- alpha * A * B + beta * C.
/// The operation can done on a contiguous subset of row indices
/// j in [idxRowCFrom, idxRowCTo[ in matrix A and C.
/// </summary>
void RnnLMTraining::MultiplyMatrixXmatrixBlas(std::vector<double> &matrixA,
                                              std::vector<double> &matrixB,
                                              std::vector<double> &matrixC,
                                              double alpha,
                                              double beta,
                                              int numRowsA,
                                              int numRowsB,
                                              int numColsC,
                                              int idxRowCFrom,
                                              int idxRowCTo) const
{
  int idxCFrom = idxRowCFrom * numColsC;
  int idxAFrom = idxRowCFrom * numRowsB;
  int heighMatrixAC = idxRowCTo - idxRowCFrom;
  double *matA = &matrixA[idxAFrom];
  double *matB = &matrixB[0];
  double *matC = &matrixC[idxCFrom];
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              heighMatrixAC, numColsC, numRowsB,
              alpha, matA, 1, matB, numColsC,
              beta, matC, numColsC);
  
}


/// <summary>
/// Matrix-matrix or vector-vector addition routine using BLAS.
/// Computes Y <- alpha * X + beta * Y.
/// </summary>
void RnnLMTraining::AddMatrixToMatrixBlas(std::vector<double> &matrixX,
                                          std::vector<double> &matrixY,
                                          double alpha,
                                          double beta,
                                          int numRows,
                                          int numCols) const
{
  double *matX = &matrixX[0];
  double *matY = &matrixY[0];
  int numElem = numRows * numCols;
  // Scale matrix Y?
  if (beta != 1.0) {
    cblas_dscal(numElem, beta, matY, 1);
  }
  cblas_daxpy(numElem, alpha, matX, 1, matY, 1);
}
