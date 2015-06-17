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
#include <algorithm>
#include <sstream>
#include <fstream>
#include <climits>

#include <math.h>
#include <time.h>
#include <assert.h>
#include "Utils.h"
#include "RnnLib.h"
#include "RnnState.h"
#include "RnnTraining.h"
#include "CorpusWordReader.h"
// Include BLAS
extern "C" {
#include <cblas.h>
}

using namespace std;


/**
 * Get the next token (word or multi-word entity) from a text file
 * and return it as an integer in the vocabulary vector.
 * Returns -1 for OOV words and -2 for end of file.
 */
int RnnLMTraining::ReadWordIndexFromFile(WordReader &reader) {
  std::string word = reader.get_next();
  // We return -2 if end of file, not to confuse with -1 for OOV words
  int index = m_eof;
  if (!(word.empty())) {
    index = m_vocab.SearchWordInVocabulary(word);
  }
  return index;
}


/**
 * Before learning the RNN model, we need to learn the vocabulary
 * from the corpus. Note that the word classes may have been initialized
 * beforehand using ReadClasses. Computes the unigram distribution
 * of words from a training file, assuming that the existing vocabulary
 * is empty.
 */
bool RnnLMTraining::LearnVocabularyFromTrainFile(int numClasses) {
  // We cannot use a class file... (classes need to be frequency-based)
  if (m_usesClassFile) {
    cerr << "Class files not implemented\n";
    return false;
  }

  // Word reader file on the training file
  Log("Reading vocabulary from file " + m_trainFile + "...\n");
  WordReader wr(m_trainFile);

  // Create an empty vocabulary structure
  Vocabulary vocab(numClasses);
  // The first word needs to be end-of-sentence
  vocab.AddWordToVocabulary("</s>");

  // Read the words in the file one by one
  long numWordsTrainingFile = 0;
  std::string nextWord = wr.get_next();
  while (!(nextWord.empty())) {
    numWordsTrainingFile++;
    // When a word is unknown, add it to the vocabulary
    vocab.AddWordToVocabulary(nextWord);
    // Read next word
    nextWord = wr.get_next();
  }
  Log("Read " + ConvString(numWordsTrainingFile) + " words\n");

  // Create the final vocabulary structure
  m_vocab = Vocabulary(numClasses);
  // The first word needs to be end-of-sentence
  m_vocab.AddWordToVocabulary("</s>");
  // Filter out words with fewer than specified mininum number of occurrences
  // and replace them by <unk>
  for (int k = 0; k < vocab.GetVocabularySize(); k++) {
    int count = vocab.m_vocabularyStorage[k].cn;
    if (count >= m_minWordOccurrences) {
      string word = vocab.m_vocabularyStorage[k].word;
      m_vocab.AddWordToVocabulary(word);
      m_vocab.SetWordCount(word, count);
    } else {
      m_vocab.AddWordToVocabulary("<unk>");
      m_oov = m_vocab.SearchWordInVocabulary("<unk>");
      int prevCount = m_vocab.m_vocabularyStorage[m_oov].cn;
      m_vocab.SetWordCount("<unk>", prevCount + count);
    }
  }

  // Simply sort the words by frequency, making sure that </s> is first
  m_vocab.SortVocabularyByFrequency();
  // Assign the words to classes
  m_vocab.AssignWordsToClasses();

  // Note the <unk> (OOV) tag
  m_oov = m_vocab.SearchWordInVocabulary("<unk>");

  m_numTrainWords = numWordsTrainingFile;
  printf("Vocab size: %d\n", GetVocabularySize());
  printf("Unknown tag at: %d\n", m_oov);
  printf("Words in train file: %ld\n", m_numTrainWords);
  return true;
}


/**
 * Once we train the RNN model, it is nice to save it to a text or binary file
 */
bool RnnLMTraining::SaveRnnModelToFile() {
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
  
  fprintf(fo, "direct connections: %d\n", GetNumDirectConnection());
  fprintf(fo, "direct order: %d\n", GetOrderDirectConnection());
  
  fprintf(fo, "bptt: %d\n", m_numBpttSteps);
  fprintf(fo, "bptt block: %d\n", m_bpttBlockSize);
  
  fprintf(fo, "vocabulary size: %d\n", GetVocabularySize());
  fprintf(fo, "class size: %d\n", GetNumClasses());
  
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
  m_vocab.Save(fo);

  int sizeHidden = GetHiddenSize();
  printf("Saving %d hidden activations...\n", sizeHidden);
  SaveBinaryVector(fo, sizeHidden, m_state.HiddenLayer);

  // Save all the weights
  m_weights.Save(fo);

  // Save the feature matrix
  if (m_featureMatrixUsed) {
    int sizeFeature = GetFeatureSize();
    printf("Saving %dx%d feature matrix...\n", sizeFeature, sizeVocabulary);
    SaveBinaryMatrix(fo, sizeFeature, sizeVocabulary, m_featureMatrix);
  }
  fclose(fo);

  return true;
}


/**
 * Cleans all activations and error vectors, in the input, hidden,
 * compression, feature and output layers, and resets word history
 */
void RnnLMTraining::ResetAllRnnActivations(RnnState &state) const {
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
  
  // Reset the vector of feature vectors
  state.FeatureLayer.assign(GetFeatureSize(), 0.0);
}


/**
 * One step of backpropagation of the errors through the RNN
 * (optionally, backpropagation through time, BPTT) and of gradient descent.
 */
void RnnLMTraining::BackPropagateErrorsThenOneStepGradientDescent(int contextWord,
                                                                  int word) {
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
  int sizeDirectConnection = GetNumDirectConnection();
  int orderDirectConnection = GetOrderDirectConnection();

  // Target word class
  int targetClass = m_vocab.WordIndex2Class(word);
  // Index at which the words in the current target word class are stored
  int idxWordClass = m_vocab.GetNthWordInClass(targetClass, 0);
  // Number of words in that class
  int numWordsInClass = m_vocab.SizeTargetClass(targetClass);

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
    int a = m_vocab.GetNthWordInClass(targetClass, c);
    m_state.OutputGradient[a] = (0 - m_state.OutputLayer[a]);
  }
  m_state.OutputGradient[word] = (1 - m_state.OutputLayer[word]);
  
  // 2) Backprop on all classes
  for (int a = sizeVocabulary; a < sizeOutput; a++) {
    m_state.OutputGradient[a] = (0 - m_state.OutputLayer[a]);
  }
  int wordClassIdx = targetClass + sizeVocabulary;
  m_state.OutputGradient[wordClassIdx] = (1 - m_state.OutputLayer[wordClassIdx]);
  
  // Reset gradients on hidden layers
  m_state.HiddenGradient.assign(sizeHidden, 0);
  m_state.CompressGradient.assign(sizeCompress, 0);
  
  // learn direct connections between words
  if (sizeDirectConnection > 0) {
    if (word != -1) {
      unsigned long long hash[c_maxNGramOrder];
      for (int a = 0; a < orderDirectConnection; a++) {
        hash[a] = 0;
      }
      for (int a = 0; a < orderDirectConnection; a++) {
        int b = 0;
        if ((a > 0) && (m_state.WordHistory[a-1] == -1)) {
          break;
        }
        hash[a] = c_Primes[0]*c_Primes[1]*(unsigned long long)(targetClass+1);
        for (b=1; b<=a; b++) {
          hash[a]+=c_Primes[(a*c_Primes[b]+b)%c_PrimesSize]*(unsigned long long)(m_state.WordHistory[b-1]+1);
        }
        hash[a] = (hash[a]%(sizeDirectConnection/2))+(sizeDirectConnection)/2;
      }
      for (int c = 0; c < numWordsInClass; c++) {
        int a = m_vocab.GetNthWordInClass(targetClass, c);
        for (int b = 0; b < orderDirectConnection; b++) {
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
    }
  }
  //
  // learn direct connections to classes
  if (sizeDirectConnection > 0) {
    unsigned long long hash[c_maxNGramOrder] = {0};
    for (int a = 0; a < orderDirectConnection; a++) {
      int b = 0;
      if (a>0) if (m_state.WordHistory[a-1] == -1) break;
      hash[a] = c_Primes[0]*c_Primes[1];
      for (b=1; b<=a; b++) {
        hash[a] += c_Primes[(a*c_Primes[b]+b)%c_PrimesSize]*(unsigned long long)(m_state.WordHistory[b-1]+1);
      }
      hash[a] = hash[a]%(sizeDirectConnection/2);
    }
    for (int a = sizeVocabulary; a < sizeOutput; a++) {
      for (int b = 0; b < orderDirectConnection; b++) {
        if (hash[b]) {
          m_weights.DirectNGram[hash[b]] +=
          alpha * m_state.OutputGradient[a] - m_weights.DirectNGram[hash[b]]*beta;
          hash[b]++;
        } else {
          break;
        }
      }
    }
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
  
  if ((sizeFeature > 0) && m_useFeatures2Output) {
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
    // If BPTT == 1, do normal BP

    // Gradient w.r.t. hidden layer
    for (int a = 0; a < sizeHidden; a++) {
      double dLdSa = m_state.HiddenLayer[a];
      m_state.HiddenGradient[a] =
      m_state.HiddenGradient[a] * dLdSa * (1 - dLdSa);
    }
    
    // Backprop and weight update hidden(t) -> input(t)
    int a = contextWord;
    if (a != -1) {
      for (int b = 0; b < sizeHidden; b++) {
        int node = a + b * sizeInput;
        m_weights.Input2Hidden[node] =
        alpha * m_state.HiddenGradient[b] * m_state.InputLayer[a]
        + coeffSGD * m_weights.Input2Hidden[node];
      }
    }
    
    // Backprop and weight update hidden(t) -> hidden(t-1)
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
    
    // Backprop and weight update hidden(t) -> feature(t)
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

    if (((m_wordCounter % m_bpttBlockSize) == 0) ||
        (m_areSentencesIndependent && (word == 0))) {
      for (int step = 0; step < m_bpttVectors.NumSteps() - 2; step++) {
        // Gradient w.r.t. hidden layer
        for (int a = 0; a < sizeHidden; a++) {
          double dLdSa = m_state.HiddenLayer[a];
          m_state.HiddenGradient[a] =
          m_state.HiddenGradient[a] * dLdSa * (1 - dLdSa);
        }

        if (sizeFeature > 0) {
          // Backprop and weight update hidden(t) -> feature(t)
          for (int b = 0; b < sizeHidden; b++) {
            for (int a = 0; a < sizeFeature; a++) {
              m_bpttVectors.WeightsFeature2Hidden[a + b * sizeFeature] +=
              alpha * m_state.HiddenGradient[b] *
              m_bpttVectors.FeatureLayer[a + step * sizeFeature];
            }
          }
        }

        // Backprop and weight update hidden -> input
        int a = m_bpttVectors.History[step];
        if (a != -1) {
          for (int b = 0; b < sizeHidden; b++)
          {
            m_bpttVectors.WeightsInput2Hidden[a + b * sizeInput] +=
            alpha * m_state.HiddenGradient[b];
          }
        }
        
        // Backprop and weight update hidden -> recurrent
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
        
        // Backpropagate error from time T-n to T-n-1
        for (int a = 0; a < sizeHidden; a++) {
          m_state.HiddenGradient[a] =
          m_state.RecurrentGradient[a] +
          m_bpttVectors.HiddenGradient[(step+1) * sizeHidden+a];
        }
        
        if (step < m_bpttVectors.NumSteps() - 3) {
          for (int a = 0; a < sizeHidden; a++) {
            m_state.HiddenLayer[a] =
            m_bpttVectors.HiddenLayer[(step+1) * sizeHidden+a];
            m_state.RecurrentLayer[a] =
            m_bpttVectors.HiddenLayer[(step+2) * sizeHidden+a];
          }
        }
      }

      // Reset BPTT accumulated gradients
      for (int a = 0; a < m_bpttVectors.NumSteps() * sizeHidden; a++) {
        m_bpttVectors.HiddenGradient[a] = 0;
      }
      
      // Restore hidden layer after BPTT
      for (int b = 0; b < sizeHidden; b++) {
        m_state.HiddenLayer[b] = m_bpttVectors.HiddenLayer[b];
      }
      
      // Weight update for recurrent weights, using BPTT accumulated gradients
      AddMatrixToMatrixBlas(m_bpttVectors.WeightsRecurrent2Hidden,
                            m_weights.Recurrent2Hidden,
                            1.0,
                            coeffSGD,
                            sizeHidden,
                            sizeHidden);
      m_bpttVectors.WeightsRecurrent2Hidden.assign(sizeHidden * sizeHidden, 0);
      
      // Weight update for feature-hidden weights, using BPTT accumulated grads
      if (sizeFeature > 0) {
        AddMatrixToMatrixBlas(m_bpttVectors.WeightsFeature2Hidden,
                              m_weights.Features2Hidden,
                              1.0,
                              coeffSGD,
                              sizeHidden,
                              sizeFeature);
        m_bpttVectors.WeightsFeature2Hidden.assign(sizeHidden * sizeFeature, 0);
      }
      
      // Weight update for input weights, using BPTT accumulated gradients
      for (int step = 0; step < m_bpttVectors.NumSteps() - 2; step++) {
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


/**
 * Train a Recurrent Neural Network model on a test file
 */
bool RnnLMTraining::TrainRnnModel() {
  // Reset the log-likelihood to ginourmous value
  double lastValidLogProbability = -1E37;
  double lastValidAccuracy = 0;
  double bestValidLogProbability = -1E37;
  double bestValidAccuracy = 0;
  // Word counter, saved at the end of last training session
  m_wordCounter = (int)m_currentPosTrainFile;
  // Keep track of the initial learning rate
  m_initialLearningRate = m_learningRate;

  // Log file
  string logFilename = m_rnnModelFile + ".log.txt";
  Log("Starting training sequential LM using file " +
      m_trainFile + "...\n", logFilename);

  // Do we use an external file with feature vectors for each
  // consecutive word in the test set?
  // Only if feature matrix (LDA/LSA topic model or Word2Vec) was not set
  // and there is a feature file
  bool isFeatureFileUsed =
  ((!m_featureMatrixUsed) && !m_featureFile.empty());
  FILE *featureFileId = NULL;
  int sizeFeature = GetFeatureSize();
  
  bool loopEpochs = true;
  while (loopEpochs) {
    // Reset the log-likelihood of the current iteration
    double trainLogProbability = 0.0;
    
    // Create a word reader on the training file
    WordReader wordReaderTrain(m_trainFile);
    // Print current epoch and learning rate
    Log("Iter: " + ConvString(m_iteration) +
        " Alpha: " + ConvString(m_learningRate) + "\n");

    // Reset everything, including word history
    ResetAllRnnActivations(m_state);
    
    // Ugly way to open the feature vector file
    if (isFeatureFileUsed) {
      featureFileId = fopen(m_featureFile.c_str(), "rb");
      int dummySizeFeature;
      fread(&dummySizeFeature, sizeof(dummySizeFeature), 1, featureFileId);
    }
    
    // Last word set to end of sentence
    int contextWord = 0;
    // Current word
    int targetWord = 0;
        
    // Start an iteration
    clock_t start = clock();
    bool loopTrain = true;
    while (loopTrain) {
      // Read next word
      targetWord = ReadWordIndexFromFile(wordReaderTrain);
      loopTrain = (targetWord > m_eof);

      if (loopTrain) {
        // Use the pre-computed feature file?
        if (isFeatureFileUsed) {
          LoadFeatureVectorAtCurrentWord(featureFileId, m_state);
        }
        // Use the topic-model features coming from a word embedding matrix?
        if (m_featureMatrixUsed) {
          UpdateFeatureVectorUsingTopicModel(contextWord, m_state);
        }
        
        // Run one step of the RNN
        ForwardPropagateOneStep(contextWord, targetWord, m_state);
        
        // For perplexity, we do not to count OOV or beginning of sentence
        if ((targetWord >= 0) && (targetWord != m_oov)) {
          // Compute the log-probability of the current word
          int targetClass = m_vocab.WordIndex2Class(targetWord);
          int outputNodeClass = targetClass + GetVocabularySize();
          double condProbaClass = m_state.OutputLayer[outputNodeClass];
          double condProbaWordGivenClass =  m_state.OutputLayer[targetWord];
          trainLogProbability +=
          log10(condProbaClass * condProbaWordGivenClass);
          m_wordCounter++;
        }
        
        // Safety check (that log-likelihood does not diverge)
        assert(!(trainLogProbability != trainLogProbability));

        // Shift memory needed for BPTT to next time step
        m_bpttVectors.Shift(contextWord);

        // Back-propagate the error and run one step of
        // stochastic gradient descent (SGD) using optional
        // back-propagation through time (BPTT)
        BackPropagateErrorsThenOneStepGradientDescent(contextWord, targetWord);
        
        // Store the current state s(t) at the end of the input layer vector
        // so that it can be used as s(t-1) at the next step
        ForwardPropagateRecurrentConnectionOnly(m_state);
        
        // Rotate the word history by one
        ForwardPropagateWordHistory(m_state, contextWord, targetWord);
        
        // Did we reach the end of the sentence?
        // If so, we need to reset the state of the neural net
        if (m_areSentencesIndependent && (targetWord == 0)) {
          ResetHiddenRnnStateAndWordHistory(m_state);
        }
      }

      // Verbose
      if ((m_wordCounter % 10000 == 0) && (m_wordCounter > 0)) {
        clock_t now = clock();
        double entropy = -trainLogProbability/log10((double)2) / m_wordCounter;
        double perplexity =
          ExponentiateBase10(-trainLogProbability / (double)m_wordCounter);
        Log("Iter," + ConvString(m_iteration) +
            ",Alpha," + ConvString(m_learningRate) +
            ",Perc," + ConvString(100 * m_wordCounter / m_numTrainWords) +
            ",TRAINent," + ConvString(entropy) +
            ",TRAINppx," + ConvString(perplexity) +
            ",words/sec," +
            ConvString(1000000 * (m_wordCounter/((double)(now-start)))) + "\n",
            logFilename);
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
    Log("Iter," + ConvString(m_iteration) +
        ",Alpha," + ConvString(m_learningRate) +
        ",Perc,100" +
        ",TRAINent," + ConvString(trainEntropy) +
        ",TRAINppx," + ConvString(trainPerplexity) +
        ",words/sec," +
        ConvString(1000000 * (m_wordCounter/((double)(now-start)))) + "\n",
        logFilename);
    
    // Validation
    vector<double> sentenceScores;
    double validLogProbability, validEntropy, validAccuracy, validPerplexity;
    TestRnnModel(m_validationFile,
                 m_featureValidationFile,
                 sentenceScores,
                 validLogProbability,
                 validPerplexity,
                 validEntropy,
                 validAccuracy);
    Log("Iter," + ConvString(m_iteration) +
        ",Alpha," + ConvString(m_learningRate) +
        ",VALIDacc," + ConvString(validAccuracy) +
        ",VALIDent," + ConvString(validEntropy) +
        ",VALIDppx," + ConvString(validPerplexity) +
        ",words/sec,0\n", logFilename);

    // Reset the position in the training file
    m_wordCounter = 0;
    m_currentPosTrainFile = 0;
    trainLogProbability = 0;
    
    // Shall we start reducing the learning rate?
    if (m_correctSentenceLabels.size() > 0) {
      // ... based on accuracy of the validation set
      if ((validAccuracy * m_minLogProbaImprovement < lastValidAccuracy)
          && (m_iteration > 4)) {
        m_doStartReducingLearningRate = true;
      }
    } else {
      // ... based on log-probability of the validation set
      if ((validLogProbability * m_minLogProbaImprovement < lastValidLogProbability)
          && (m_iteration > 4)) {
        m_doStartReducingLearningRate = true;
      }
    }
    if (m_doStartReducingLearningRate) {
      m_learningRate /= 1.5;
    }
    // We need to stop at some point!
    if (m_learningRate < 0.0001) {
      loopEpochs = false;
    }

    if (loopEpochs) {
      // Store last value of accuracy and log-probability
      lastValidLogProbability = validLogProbability;
      lastValidAccuracy = validAccuracy;
      validLogProbability = 0;
      m_iteration++;
      // Save the best model
      if (validAccuracy > bestValidAccuracy) {
        SaveRnnModelToFile();
        SaveWordEmbeddings(m_rnnModelFile + ".word_embeddings.txt");
        Log("Saved the best model so far\n");
        bestValidAccuracy = validAccuracy;
        bestValidLogProbability = validLogProbability;
      }
    }
  }

  return true;
}


/**
 * Test a Recurrent Neural Network model on a test file
 */
bool RnnLMTraining::TestRnnModel(const string &testFile,
                                 const string &featureFile,
                                 vector<double> &sentenceScores,
                                 double &logProbability,
                                 double &perplexity,
                                 double &entropy,
                                 double &accuracy) {
  Log("RnnTrainingLM::testNet()\n");

  // Scores file
  string scoresFilename = m_rnnModelFile + ".scores.";
  size_t sep = testFile.find_last_of("\\/");
  if (sep != string::npos)
    scoresFilename += testFile.substr(sep + 1, testFile.size() - sep - 1);
  scoresFilename += ".iter" + ConvString(m_iteration) + ".txt";
  Log("Writing sentence scores to " + scoresFilename + "...\n");

  // Do we use an external file with feature vectors for each
  // consecutive word in the test set?
  // Only if feature matrix (LDA/LSA topic model or Word2Vec) was not set
  // and there is a feature file
  bool isFeatureFileUsed =
  ((!m_featureMatrixUsed) && !featureFile.empty());
  FILE *featureFileId = NULL;
  int sizeFeature = GetFeatureSize();
  // Ugly way to open the feature vector file
  if (isFeatureFileUsed) {
    if (featureFile.empty()) {
      printf("Feature file for the test data is needed to evaluate this model (use -features <FILE>)\n");
      return false;
    }
    featureFileId = fopen(featureFile.c_str(), "rb");
    int a;
    fread(&a, sizeof(a), 1, featureFileId);
    if (a != sizeFeature) {
      printf("Mismatch between feature vector size in model file and feature file (model uses %d features, in %s found %d features)\n", sizeFeature, m_featureFile.c_str(), a);
      return false;
    }
  }

  // This function does what ResetHiddenRnnStateAndWordHistory does
  // and also resets the features, inputs, outputs and compression layer
  ResetAllRnnActivations(m_state);
  
  // Create a word reader on the test file
  WordReader wordReaderTest(testFile);
  
  // Last word set to end of sentence
  int contextWord = 0;
  // Reset the log-likelihood
  logProbability = 0.0;
  double sentenceLogProbability = 0.0;
  // Reset the word counter
  int uniqueWordCounter = 0;
  int numUnk = 0;
  // Reset the sentence scores
  sentenceScores.clear();
  
  // Since we just set s(1)=0, this will set the state s(t-1) to 0 as well...
  ForwardPropagateRecurrentConnectionOnly(m_state);
  if (m_areSentencesIndependent) {
    ResetHiddenRnnStateAndWordHistory(m_state);
  }
  
  // Iterate over the test file
  bool loopTest = true;
  while (loopTest) {
    // Get the index of the next word (or -1 if OOV or -2 if end of file)
    int targetWord = ReadWordIndexFromFile(wordReaderTest);
    loopTest = (targetWord > m_eof);
    
    if (loopTest) {
      // Use the pre-computed feature file?
      if (isFeatureFileUsed) {
        LoadFeatureVectorAtCurrentWord(featureFileId, m_state);
      }
      // Use the topic-model features coming from a word embedding matrix?
      if (m_featureMatrixUsed) {
        UpdateFeatureVectorUsingTopicModel(contextWord, m_state);
      }
      
      // Run one step of the RNN
      ForwardPropagateOneStep(contextWord, targetWord, m_state);
      
      // For perplexity, we do not count OOV words and beginning of sentence...
      if ((targetWord >= 0) && (targetWord != m_oov)) {
        // Compute the log-probability of the current word
        int targetClass = m_vocab.WordIndex2Class(targetWord);
        int outputNodeClass = targetClass + GetVocabularySize();
        double condProbaClass = m_state.OutputLayer[outputNodeClass];
        double condProbaWordGivenClass =  m_state.OutputLayer[targetWord];
        double logProbabilityWord =
        log10(condProbaClass * condProbaWordGivenClass);
        logProbability += logProbabilityWord;
        sentenceLogProbability += logProbabilityWord;
        uniqueWordCounter++;

        // Verbose
        if (m_debugMode) {
          Log(ConvString(targetWord) + "\t" +
              ConvString(logProbabilityWord) + "\t" +
              m_vocab.Word2WordIndex(contextWord) + "\t" +
              m_vocab.Word2WordIndex(targetWord) + "\t" +
              ConvString(m_vocab.WordIndex2Class(targetWord)) + "\t" +
              ConvString(m_vocab.WordIndex2Class(contextWord)) + "\n");
        }
      } else {
        if (m_debugMode) {
          // Out-of-vocabulary words have probability 0 and index -1
          Log("-1\t0\t" +
              m_vocab.Word2WordIndex(contextWord) + "\t" +
              m_vocab.Word2WordIndex(targetWord) + "\t-1\t-1\n");
        }
        numUnk++;
      }
      
      // Store the current state s(t) at the end of the input layer vector
      // so that it can be used as s(t-1) at the next step
      ForwardPropagateRecurrentConnectionOnly(m_state);
      
      // Rotate the word history by one
      ForwardPropagateWordHistory(m_state, contextWord, targetWord);
      
      // Did we reach the end of the sentence?
      // If so, we need to reset the state of the neural net
      // and to save the current sentence score
      if (m_areSentencesIndependent && (targetWord == 0)) {
        ResetHiddenRnnStateAndWordHistory(m_state);
        sentenceScores.push_back(sentenceLogProbability);
        // Write the sentence score to a file
        Log(ConvString(sentenceLogProbability) + "\n", scoresFilename);
        sentenceLogProbability = 0.0;
      }
    }
  }
  
  if (isFeatureFileUsed) {
    fclose(featureFileId);
  }
  
  // Log file
  string logFilename = m_rnnModelFile + ".test.log.txt";

  // Return the total logProbability
  Log("Log probability: " + ConvString(logProbability) +
      ", number of words " + ConvString(uniqueWordCounter) +
      " (" + ConvString(numUnk) + " <unk>," +
      " " + ConvString(sentenceScores.size()) + " sentences)\n", logFilename);

  // Compute the perplexity and entropy
  perplexity = (uniqueWordCounter == 0) ? 0 :
  ExponentiateBase10(-logProbability / (double)uniqueWordCounter);
  entropy = (uniqueWordCounter == 0) ? 0 :
  -logProbability / log10((double)2) / uniqueWordCounter;
  Log("PPL net (perplexity without OOV): " + ConvString(perplexity) + "\n",
      logFilename);

  // Load the labels
  LoadCorrectSentenceLabels(m_fileCorrectSentenceLabels);
  // Compute the accuracy
  accuracy = AccuracyNBestList(sentenceScores, m_correctSentenceLabels);
  Log("Accuracy: " + ConvString(accuracy * 100) + "% on " +
      ConvString(sentenceScores.size()) + " sentences\n", logFilename);

  return true;
}


/**
 * Load a file containing the classification labels
 */
void RnnLMTraining::LoadCorrectSentenceLabels(const std::string &labelFile) {
  m_correctSentenceLabels.clear();
  ifstream file(labelFile, ifstream::in);
  int label = 0;
  while (file >> label) {
    m_correctSentenceLabels.push_back(label);
  }
  Log("Loaded correct labels for " + ConvString(m_correctSentenceLabels.size()) +
      " validation/test sentences\n");
  file.close();
}


/**
 * Compute the accuracy of selecting the top candidate (based on score)
 * among n-best lists
 */
double RnnLMTraining::AccuracyNBestList(std::vector<double> scores,
                                        std::vector<int> &correctLabels) const {
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


/**
 * Read the feature vector for the current word
 * in the train/test/valid file and update the feature vector
 * in the state
 * TODO: convert to ifstream
 */
bool RnnLMTraining::LoadFeatureVectorAtCurrentWord(FILE *f,
                                                   RnnState &state) {
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


/**
 * Simply write the word projections/embeddings to a text file.
 */
void RnnLMTraining::SaveWordEmbeddings(const string &filename) {
  FILE *fid = fopen(filename.c_str(), "wb");
  
  fprintf(fid, "%d %d\n", GetVocabularySize(), GetHiddenSize());
  
  for (int a = 0; a < GetVocabularySize(); a++) {
    fprintf(fid, "%s ", m_vocab.GetNthWord(a).c_str());
    for (int b = 0; b < GetHiddenSize(); b++) {
      fprintf(fid, "%lf ", m_weights.Input2Hidden[a + b * GetInputSize()]);
    }
    fprintf(fid, "\n");
  }
  
  fclose(fid);
}


/**
 * Matrix-vector multiplication routine, somewhat accelerated using loop
 * unrolling over 8 registers. Computes x <- x + A' * y,
 * i.e., the "inverse" operation to y = A * x (adding the result to x)
 * where A is of size N x M, x is of length M and y is of length N.
 * The operation can done on a contiguous subset of indices
 * j in [idxYFrom, idxYTo[ of vector y.
 */
void RnnLMTraining::GradientMatrixXvectorBlas(vector<double> &vectorX,
                                              vector<double> &vectorY,
                                              vector<double> &matrixA,
                                              int widthMatrix,
                                              int idxYFrom,
                                              int idxYTo) const {
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


/**
 * Matrix-matrix multiplication routine using BLAS.
 * Computes C <- alpha * A * B + beta * C.
 * The operation can done on a contiguous subset of row indices
 * j in [idxRowCFrom, idxRowCTo[ in matrix A and C.
 */
void RnnLMTraining::MultiplyMatrixXmatrixBlas(std::vector<double> &matrixA,
                                              std::vector<double> &matrixB,
                                              std::vector<double> &matrixC,
                                              double alpha,
                                              double beta,
                                              int numRowsA,
                                              int numRowsB,
                                              int numColsC,
                                              int idxRowCFrom,
                                              int idxRowCTo) const {
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


/**
 * Matrix-matrix or vector-vector addition routine using BLAS.
 * Computes Y <- alpha * X + beta * Y.
 */
void RnnLMTraining::AddMatrixToMatrixBlas(std::vector<double> &matrixX,
                                          std::vector<double> &matrixY,
                                          double alpha,
                                          double beta,
                                          int numRows,
                                          int numCols) const {
  double *matX = &matrixX[0];
  double *matY = &matrixY[0];
  int numElem = numRows * numCols;
  // Scale matrix Y?
  if (beta != 1.0) {
    cblas_dscal(numElem, beta, matY, 1);
  }
  cblas_daxpy(numElem, alpha, matX, 1, matY, 1);
}
