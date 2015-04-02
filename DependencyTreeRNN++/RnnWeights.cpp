//
//  RnnWeights.cpp
//  DependencyTreeRNN
//
//  Created by Piotr Mirowski on 15/03/2015.
//  Copyright (c) 2015 Piotr Mirowski. All rights reserved.
//

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
#ifdef USE_HASHTABLES
  DirectBiGram.clear();
  DirectTriGram.clear();
#else
  DirectNGram.assign(m_sizeDirectConnection, 0.0);
#endif
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
#ifdef USE_HASHTABLES
  DirectBiGram.clear();
  DirectTriGram.clear();
#else
  DirectNGram.clear();
#endif
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
#ifdef USE_HASHTABLES
#else
    // Read the direct connections
    ReadBinaryVector(fi, m_sizeDirectConnection, DirectNGram);
#endif
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
#ifdef USE_HASHTABLES
#else
    for (long long aa = 0; aa < m_sizeDirectConnection; aa++) {
      float fl = (float)(DirectNGram[aa]);
      fwrite(&fl, 4, 1, fo);
    }
#endif
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
