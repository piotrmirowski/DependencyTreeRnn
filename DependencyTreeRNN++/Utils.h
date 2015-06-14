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

#ifndef DependencyTreeRNN_Utils_h
#define DependencyTreeRNN_Utils_h

#include <stdio.h>
#include <vector>

#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>


/**
 * Log to screen and to file (append)
 */
static void Log(std::string str, std::string logFilename) {
  std::ostringstream buf;
  std::ofstream logFile(logFilename, std::fstream::app);
  buf << str;
  logFile << buf.str() << std::flush;
  std::cout << buf.str() << std::flush;
  buf.str("");
  buf.clear();
}


/**
 * Log to screen only
 */
static void Log(std::string str) {
  std::ostringstream buf;
  buf << str;
  std::cout << buf.str() << std::flush;
  buf.str("");
  buf.clear();
}


/**
 * Read a matrix of floats in binary format
 */
static void ReadBinaryMatrix(FILE *fi, int sizeIn, int sizeOut,
                             std::vector<double> &vec) {
  if (sizeIn * sizeOut == 0) {
    return;
  }
  for (int idxOut = 0; idxOut < sizeOut; idxOut++) {
    for (int idxIn = 0; idxIn < sizeIn; idxIn++) {
      float val;
      fread(&val, 4, 1, fi);
      vec[idxIn + idxOut * sizeIn] = val;
    }
  }
}


/**
 * Read a vector of floats in binary format
 */
static void ReadBinaryVector(FILE *fi, long long size,
                             std::vector<double> &vec) {
  for (long long aa = 0; aa < size; aa++) {
    float val;
    fread(&val, 4, 1, fi);
    vec[aa] = val;
  }
}


/**
 * Save a matrix of floats in binary format
 */
static void SaveBinaryMatrix(FILE *fo, int sizeIn, int sizeOut,
                             const std::vector<double> &vec) {
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


/**
 * Save a vector of floats in binary format
 */
static void SaveBinaryVector(FILE *fo, long long size,
                             const std::vector<double> &vec) {
  for (long long aa = 0; aa < size; aa++) {
    float val = vec[aa];
    fwrite(&val, 4, 1, fo);
  }
}


/**
 * Random number generator of double random number in range [min, max]
 */
static double GenerateUniformRandomNumber(double min, double max) {
  return rand() / ((double)RAND_MAX) * (max - min) + min;
}


/**
 * Random number generator (approximate Gaussian distribution),
 * zero-mean and standard deviation 0.1
 */
static double GenerateNormalRandomNumber() {
  return (GenerateUniformRandomNumber(-0.1, 0.1)
          + GenerateUniformRandomNumber(-0.1, 0.1)
          + GenerateUniformRandomNumber(-0.1, 0.1));
}


/**
 * Randomize a vector with small numbers to get zero-mean random numbers
 */
static void RandomizeVector(std::vector<double> &vec) {
  for (size_t k = 0; k < vec.size(); k++) {
    vec[k] = GenerateNormalRandomNumber();
  }
}


/**
 * Convert int or double to string
 */
static std::string ConvString(int val) {
  return std::to_string(static_cast<long long int>(val));
}


/**
 * Convert int or double to string
 */
static std::string ConvString(size_t val) {
  return std::to_string(static_cast<long long int>(val));
}


/**
 * Convert int or double to string
 */
static std::string ConvString(long int val) {
  return std::to_string(static_cast<long long int>(val));
}


/**
 * Convert int or double to string
 */
static std::string ConvString(long long int val) {
  return std::to_string(static_cast<long long int>(val));
}


/**
 * Convert int or double to string
 */
static std::string ConvString(double val) {
  return std::to_string(static_cast<long double>(val));
}


#endif
