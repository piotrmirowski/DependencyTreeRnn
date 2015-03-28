//
//  Utils.h
//  DependencyTreeRNN
//
//  Created by Piotr Mirowski on 03/03/2015.
//  Copyright (c) 2015 Piotr Mirowski. All rights reserved.
//

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

#endif
