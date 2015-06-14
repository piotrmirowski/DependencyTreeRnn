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
#include <vector>
#include <iostream>
#include <sstream>
#include <assert.h>
#include "Utils.h"
#include "RnnWeights.h"

using namespace std;

/**
 * Constructor
 */
RnnWeights::RnnWeights(int sizeVocabulary,
                       int sizeHidden,
                       int sizeFeature,
                       int sizeClasses,
                       int sizeCompress,
                       long long sizeDirectConnection)
: m_sizeVocabulary(sizeVocabulary),
m_sizeHidden(sizeHidden),
m_sizeFeature(sizeFeature),
m_sizeClasses(sizeClasses),
m_sizeCompress(sizeCompress),
m_sizeDirectConnection(sizeDirectConnection),
m_sizeInput(sizeVocabulary),
m_sizeOutput(sizeVocabulary + sizeClasses) {

  // Sanity check
  assert(sizeClasses <= sizeVocabulary);
  cout << "RnnWeights: allocate " << m_sizeInput << " inputs ("
  << sizeVocabulary << " words), "
  << m_sizeClasses << " classes, "
  << m_sizeHidden << " hiddens, "
  << m_sizeFeature << " features, "
  << m_sizeCompress << " compressed, "
  << m_sizeDirectConnection << " n-grams\n";

  // Allocate the weights connecting those layers
  // (will be assigned random values later)
  Input2Hidden.resize(m_sizeInput * m_sizeHidden);
  Recurrent2Hidden.resize(m_sizeHidden * m_sizeHidden);
  Features2Hidden.resize(m_sizeFeature * m_sizeHidden);
  Features2Output.resize(m_sizeFeature * m_sizeOutput);
  if (sizeCompress == 0) {
    Hidden2Output.resize(m_sizeHidden * m_sizeOutput);
  } else {
    // Add a compression layer between hidden nodes and outputs
    Hidden2Output.resize(m_sizeHidden * m_sizeCompress);
    Compress2Output.resize(m_sizeCompress * m_sizeOutput);
  }
  // Change that to proper normal distribution
  // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  RandomizeVector(Input2Hidden);
  RandomizeVector(Recurrent2Hidden);
  if (sizeFeature > 0) {
    RandomizeVector(Features2Hidden);
    RandomizeVector(Features2Output);
  }
  if (sizeCompress > 0) {
    RandomizeVector(Compress2Output);
  }
  RandomizeVector(Hidden2Output);

  // Initialize the direct n-gram connections
  DirectNGram.assign(m_sizeDirectConnection, 0.0);
} // RnnWeights()


/**
 * Clear all the weights (before loading a new copy), to save memory
 */
void RnnWeights::Clear() {
  Input2Hidden.clear();
  Recurrent2Hidden.clear();
  Features2Hidden.clear();
  Features2Output.clear();
  if (m_sizeCompress == 0) {
    Hidden2Output.clear();
  } else {
    Hidden2Output.clear();
    Compress2Output.clear();
  }
  DirectNGram.clear();
}


/**
 * Load the weights matrices from a file
 */
void RnnWeights::Load(FILE *fi) {
  // Read the weights of input -> hidden connections
  Log("Reading " + ConvString(m_sizeHidden) +
      "x" + ConvString(m_sizeInput) + " input->hidden weights...\n");
  ReadBinaryMatrix(fi, m_sizeInput, m_sizeHidden, Input2Hidden);
  // Read the weights of recurrent hidden -> hidden connections
  Log("Reading " + ConvString(m_sizeHidden) + "x" + ConvString(m_sizeHidden) +
      " recurrent hidden->hidden weights...\n");
  ReadBinaryMatrix(fi, m_sizeHidden, m_sizeHidden, Recurrent2Hidden);
  // Read the weights of feature -> hidden connections
  Log("Reading " + ConvString(m_sizeHidden) + "x" + ConvString(m_sizeFeature) +
      " feature->hidden weights...\n");
  ReadBinaryMatrix(fi, m_sizeFeature, m_sizeHidden, Features2Hidden);
  // Read the weights of feature -> output connections
  Log("Reading " + ConvString(m_sizeOutput) + "x" + ConvString(m_sizeFeature) +
      " feature->output weights...\n");
  ReadBinaryMatrix(fi, m_sizeFeature, m_sizeOutput, Features2Output);
  if (m_sizeCompress == 0) {
    // Read the weights of hidden -> output connections
    Log("Reading " + ConvString(m_sizeOutput) + "x" + ConvString(m_sizeHidden) +
        " hidden->output weights...\n");
    ReadBinaryMatrix(fi, m_sizeHidden, m_sizeOutput, Hidden2Output);
  } else {
    // Read the weights of hidden -> compression connections
    Log("Reading " + ConvString(m_sizeCompress) + "x" + ConvString(m_sizeHidden) +
        " hidden->compress weights...\n");
    ReadBinaryMatrix(fi, m_sizeHidden, m_sizeCompress, Hidden2Output);
    // Read the weights of compression -> output connections
    Log("Reading " + ConvString(m_sizeOutput) + "x" + ConvString(m_sizeCompress) +
        " compress->output weights...\n");
    ReadBinaryMatrix(fi, m_sizeCompress, m_sizeOutput, Compress2Output);
  }
  if (m_sizeDirectConnection > 0) {
    Log("Reading " + ConvString(m_sizeDirectConnection) +
        " n-gram connections...\n");
    // Read the direct connections
    ReadBinaryVector(fi, m_sizeDirectConnection, DirectNGram);
  }
} // void Load()


/**
 * Save the weights matrices to a file
 */
void RnnWeights::Save(FILE *fo) {
  string logFilename = "log_saving.txt";
  // Save the weights U: input -> hidden (i.e., the word embeddings)
  Log("Saving " + ConvString(m_sizeHidden) + "x" + ConvString(m_sizeInput) +
      " input->hidden weights...\n", logFilename);
  SaveBinaryMatrix(fo, m_sizeInput, m_sizeHidden, Input2Hidden);
  // Save the weights W: recurrent hidden -> hidden (i.e., the time-delay)
  Log("Saving " + ConvString(m_sizeHidden) + "x" + ConvString(m_sizeHidden) +
      " recurrent hidden->hidden weights...\n", logFilename);
  SaveBinaryMatrix(fo, m_sizeHidden, m_sizeHidden, Recurrent2Hidden);
  // Save the weights feature -> hidden
  Log("Saving " + ConvString(m_sizeHidden) + "x" + ConvString(m_sizeFeature) +
      " feature->hidden weights...\n", logFilename);
  SaveBinaryMatrix(fo, m_sizeFeature, m_sizeHidden, Features2Hidden);
  // Save the weights G: feature -> output
  Log("Saving " + ConvString(m_sizeOutput) + "x" + ConvString(m_sizeFeature) +
      " feature->output weights...\n", logFilename);
  SaveBinaryMatrix(fo, m_sizeFeature, m_sizeOutput, Features2Output);
  // Save the weights hidden -> compress and compress -> output
  // or simply the weights V: hidden -> output
  if (m_sizeCompress > 0) {
    Log("Saving " + ConvString(m_sizeCompress) + "x" + ConvString(m_sizeHidden) +
        " hidden->compress weights...\n", logFilename);
    SaveBinaryMatrix(fo, m_sizeHidden, m_sizeCompress, Hidden2Output);
    Log("Saving " + ConvString(m_sizeOutput) + "x" + ConvString(m_sizeCompress) +
        " compress->output weights...\n", logFilename);
    SaveBinaryMatrix(fo, m_sizeCompress, m_sizeOutput, Compress2Output);
  } else {
    Log("Saving " + ConvString(m_sizeOutput) + "x" + ConvString(m_sizeHidden) +
        " hidden->output weights...\n", logFilename);
    SaveBinaryMatrix(fo, m_sizeHidden, m_sizeOutput, Hidden2Output);
  }
  if (m_sizeDirectConnection > 0) {
    // Save the direct connections
    Log("Saving " + ConvString(m_sizeDirectConnection) +
        " n-gram connections...\n", logFilename);
    for (long long aa = 0; aa < m_sizeDirectConnection; aa++) {
      float fl = (float)(DirectNGram[aa]);
      fwrite(&fl, 4, 1, fo);
    }
  }
} // void Save()


/**
 * Debug function
 */
void RnnWeights::Debug() {
  Log("input2hidden: " + ConvString(m_sizeInput) + "x" +
      ConvString(m_sizeHidden) + " " +
      ConvString(Input2Hidden[(m_sizeInput-1)*(m_sizeHidden-1)]) + "\n");
  Log("recurrent2hidden: " + ConvString(m_sizeHidden) + "x" +
      ConvString(m_sizeHidden) + " " +
      ConvString(Recurrent2Hidden[(m_sizeHidden-1)*(m_sizeHidden-1)]) + "\n");
  Log("hidden2output: " + ConvString(m_sizeHidden) + "x" +
      ConvString(m_sizeOutput) + " " +
      ConvString(Hidden2Output[(m_sizeOutput-1)*(m_sizeHidden-1)]) + "\n");
  if (m_sizeFeature > 0) {
    Log("features2hidden: " + ConvString(m_sizeFeature) + "x" +
        ConvString(m_sizeHidden) + " " +
        ConvString(Features2Hidden[(m_sizeFeature-1)*(m_sizeHidden-1)]) + "\n");
    Log("features2output: " + ConvString(m_sizeFeature) + "x" +
        ConvString(m_sizeOutput) + " " +
        ConvString(Features2Output[(m_sizeFeature-1)*(m_sizeOutput-1)]) + "\n");
  }
  if (m_sizeDirectConnection > 0)
    Log("direct: " + ConvString(m_sizeDirectConnection) + " " +
      ConvString(DirectNGram[m_sizeDirectConnection-1]) + "\n");
} // void Debug()
