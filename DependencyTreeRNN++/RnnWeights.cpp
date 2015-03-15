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
#include <assert.h>
#include "Utils.h"
#include "RnnWeights.h"


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
  std::cout << "RnnWeights: allocate " << m_sizeInput << " inputs ("
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
 * Load the weights matrices from a file
 */
void RnnWeights::Load(FILE *fi) {
  // Read the weights of input -> hidden connections
  ReadBinaryMatrix(fi, m_sizeInput, m_sizeHidden, Input2Hidden);
  // Read the weights of recurrent hidden -> hidden connections
  ReadBinaryMatrix(fi, m_sizeHidden, m_sizeHidden, Recurrent2Hidden);
  // Read the weights of feature -> hidden connections
  ReadBinaryMatrix(fi, m_sizeFeature, m_sizeHidden, Features2Hidden);
  // Read the weights of feature -> output connections
  ReadBinaryMatrix(fi, m_sizeFeature, m_sizeOutput, Features2Output);
  if (m_sizeCompress == 0) {
    // Read the weights of hidden -> output connections
    ReadBinaryMatrix(fi, m_sizeHidden, m_sizeOutput, Hidden2Output);
  } else {
    // Read the weights of hidden -> compression connections
    ReadBinaryMatrix(fi, m_sizeHidden, m_sizeCompress, Hidden2Output);
    // Read the weights of compression -> output connections
    ReadBinaryMatrix(fi, m_sizeCompress, m_sizeOutput, Compress2Output);
  }
  if (m_sizeDirectConnection > 0) {
#ifdef USE_HASHTABLES
#else
    // Read the direct connections
    ReadBinaryVector(fi, m_sizeDirectConnection, DirectNGram);
#endif
  }
  Debug();
} // void Load()


/**
 * Save the weights matrices to a file
 */
void RnnWeights::Save(FILE *fo) {
  // Save the weights U: input -> hidden (i.e., the word embeddings)
  printf("Saving %dx%d input->hidden weights...\n", m_sizeHidden, m_sizeInput);
  SaveBinaryMatrix(fo, m_sizeInput, m_sizeHidden, Input2Hidden);
  // Save the weights W: recurrent hidden -> hidden (i.e., the time-delay)
  printf("Saving %dx%d recurrent hidden->hidden weights...\n",
         m_sizeHidden, m_sizeHidden);
  SaveBinaryMatrix(fo, m_sizeHidden, m_sizeHidden, Recurrent2Hidden);
  // Save the weights feature -> hidden
  printf("Saving %dx%d feature->hidden weights...\n", m_sizeHidden, m_sizeFeature);
  SaveBinaryMatrix(fo, m_sizeFeature, m_sizeHidden, Features2Hidden);
  // Save the weights G: feature -> output
  printf("Saving %dx%d feature->output weights...\n", m_sizeOutput, m_sizeFeature);
  SaveBinaryMatrix(fo, m_sizeFeature, m_sizeOutput, Features2Output);
  // Save the weights hidden -> compress and compress -> output
  // or simply the weights V: hidden -> output
  if (m_sizeCompress > 0) {
    printf("Saving %dx%d hidden->compress weights...\n", m_sizeCompress, m_sizeHidden);
    SaveBinaryMatrix(fo, m_sizeHidden, m_sizeCompress, Hidden2Output);
    printf("Saving %dx%d compress->output weights...\n", m_sizeCompress, m_sizeOutput);
    SaveBinaryMatrix(fo, m_sizeCompress, m_sizeOutput, Compress2Output);
  } else {
    printf("Saving %dx%d hidden->output weights...\n", m_sizeOutput, m_sizeHidden);
    SaveBinaryMatrix(fo, m_sizeHidden, m_sizeOutput, Hidden2Output);
  }
  if (m_sizeDirectConnection > 0) {
    // Save the direct connections
    printf("Saving %lld n-gram connections...\n", m_sizeDirectConnection);
#ifdef USE_HASHTABLES
#else
    for (long long aa = 0; aa < m_sizeDirectConnection; aa++) {
      float fl = (float)(DirectNGram[aa]);
      fwrite(&fl, 4, 1, fo);
    }
#endif
  }
  Debug();
} // void Save()


/**
 * Debug function
 */
void RnnWeights::Debug() {
  std::cout << "input2hidden: " << m_sizeInput << " " << m_sizeHidden << " "
  << Input2Hidden[100] << std::endl;
  std::cout << "recurrent2hidden: " << m_sizeHidden << " " << m_sizeHidden << " "
  << Recurrent2Hidden[100] << std::endl;
  std::cout << "hidden2output: " << m_sizeHidden << " " << m_sizeOutput << " "
  << Hidden2Output[100] << std::endl;
  std::cout << "features2hidden: " << m_sizeFeature << " " << m_sizeHidden << " "
  << Features2Hidden[100] << std::endl;
  std::cout << "features2output: " << m_sizeFeature << " " << m_sizeOutput << " "
  << Features2Output[100] << std::endl;
  if (m_sizeDirectConnection > 0)
  std::cout << "direct: " << m_sizeDirectConnection << " "
  << DirectNGram[100] << std::endl;
} // void Debug()
