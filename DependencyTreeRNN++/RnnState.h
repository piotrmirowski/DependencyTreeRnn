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

#ifndef DependencyTreeRNN___RnnState_h
#define DependencyTreeRNN___RnnState_h

#include <vector>

/// <summary>
/// State vectors in the RNN model, storing per-word and per-class activations
/// </summary>
struct RnnState
{
    // Input layer (i.e., words)
    std::vector<double> InputLayer;
    // Input feature layer (e.g., topics)
    std::vector<double> FeatureLayer;
    // Hidden layer at previous time step
    std::vector<double> RecurrentLayer;
    // Hidden layer
    std::vector<double> HiddenLayer;
    // Second (compression) hidden layer
    std::vector<double> CompressLayer;
    // Output layer
    std::vector<double> OutputLayer;
    
    // Gradient to the words in input layer
    std::vector<double> InputGradient;
    // Gradient to the features in input layer
    std::vector<double> FeatureGradient;
    // Gradient to the hidden state at previous time step
    std::vector<double> RecurrentGradient;
    // Gradient to the hidden layer
    std::vector<double> HiddenGradient;
    // Gradient to the second (compression) hidden layer
    std::vector<double> CompressGradient;
    // Gradient to the output layer
    std::vector<double> OutputGradient;
    
    // Word history
    std::vector<int> WordHistory;
};


struct RnnBptt
{
    std::vector<int> History;
    std::vector<double> HiddenLayer;
    std::vector<double> HiddenGradient;
    std::vector<double> FeatureLayer;
    std::vector<double> FeatureGradient;
    std::vector<double> WeightsInput2Hidden;
    std::vector<double> WeightsRecurrent2Hidden;
    std::vector<double> WeightsFeature2Hidden;
};

#endif
