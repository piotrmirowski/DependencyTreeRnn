// Copyright (c) 2014-2015 Piotr Mirowski. All rights reserved.
//                         piotr.mirowski@computer.org
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <map>
#include <iostream>
#include <sstream>
#include <assert.h>
#include "ReadJson.h"
#include "RnnState.h"
#include "CorpusUnrollsReader.h"
#include "RnnDependencyTreeLib.h"

// Include BLAS
#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif


/// <summary>
/// Before learning the RNN model, we need to learn the vocabulary
/// from the corpus. Note that the word classes may have been initialized
/// beforehand using ReadClasses. Computes the unigram distribution
/// of words from a training file, assuming that the existing vocabulary
/// is empty.
/// </summary>
bool RnnTreeLM::LearnVocabularyFromTrainFile(int numClasses) {

  // We cannot use a class file... (classes need to be frequency-based)
  if (m_usesClassFile) {
    cerr << "Class files not implemented\n";
    return false;
  }

  // Read the vocabulary from all the files
  // OOV <unk> and EOS </s> tokens are added automatically.
  // Also count the number of words in all the books.
  m_numTrainWords =
  m_corpusVocabulary.ReadVocabulary(m_typeOfDepLabels == 1);
  printf("Words in train file: %ld\n", m_numTrainWords);

  // Filter the vocabulary based on frequency
  // and sort it based on frequency
  m_corpusTrain.FilterSortVocabulary(m_corpusVocabulary);
  
  // Print the vocabulary size and store the file size in words
  printf("Vocab size (before pruning): %d\n",
         m_corpusVocabulary.NumWords());
  printf("Vocab size (after pruning): %d\n",
         m_corpusTrain.NumWords());
  printf("Label vocab size: %d\n",
         m_corpusTrain.NumLabels());

  // Copy the vocabulary to the other corpus
  m_corpusValidTest.CopyVocabulary(m_corpusTrain);

  // Export the vocabulary
  m_corpusTrain.ExportVocabulary(m_rnnModelFile + ".vocab.txt");

  // Assign the vocabulary to the model
  return AssignVocabularyFromCorpora(numClasses);
}


/// <summary>
/// Before learning the RNN model, we need to learn the vocabulary
/// from the corpus. Note that the word classes may have been initialized
/// beforehand using ReadClasses. Computes the unigram distribution
/// of words from a training file, assuming that the existing vocabulary
/// is empty.
/// </summary>
bool RnnTreeLM::AssignVocabularyFromCorpora(int numClasses) {

  // Create an empty vocabulary structure for words
  m_vocab = Vocabulary(numClasses);
  // The first word needs to be end-of-sentence? TBD...
  m_vocab.AddWordToVocabulary("</s>");
  // Copy the words currently in the corpus
  // and insert them into the vocabulary of the RNN
  // and to the maps: word <-> index
  for (int k = 0; k < m_corpusTrain.NumWords(); k++) {
    // Get the word
    string word = m_corpusTrain.vocabularyReverse[k];
    // Lookup it up in the vocabulary and add to vocabulary if required
    m_vocab.AddWordToVocabulary(word);
    // Store the count of words in the vocabulary
    double count = m_corpusTrain.wordCountsDiscounted[k];
    m_vocab.SetWordCount(word, (int)round(count));
  }
  // Note that we do not sort the words by frequency, as they are already sorted

  // Assign the words to classes
  m_vocab.AssignWordsToClasses();

  // Note the <unk> (OOV) tag
  m_oov = m_vocab.SearchWordInVocabulary("<unk>");

  // Create an empty vocabulary structure for labels
  m_labels = Vocabulary(1);
  // Copy the labels currently in the corpus
  for (int k = 0; k < m_corpusTrain.NumLabels(); k++) {
    // Get the word
    string label = m_corpusTrain.labelsReverse[k];
    // Lookup it up in the vocabulary of labels and add it if needed
    m_labels.AddWordToVocabulary(label);
  }

  printf("Vocab size: %d\n", GetVocabularySize());
  printf("Unknown tag at: %d\n", m_oov);
  printf("Label vocab size: %d\n", GetLabelSize());
  return true;
}


/// <summary>
/// Reset the vector of feature labels
/// </summary>
void RnnTreeLM::ResetFeatureLabelVector(RnnState &state) const {
  state.FeatureLayer.assign(GetFeatureSize(), 0.0);
}


/// <summary>
/// Update the vector of feature labels
/// </summary>
void RnnTreeLM::UpdateFeatureLabelVector(int label, RnnState &state) const {
  // Time-decay the previous labels using weight gamma
  int sizeFeatures = GetFeatureSize();
  for (int a = 0; a < sizeFeatures; a++) {
    state.FeatureLayer[a] *= m_featureGammaCoeff;
  }
  // Find the current label and set it to 1
  if ((label >= 0) && (label < sizeFeatures)) {
    state.FeatureLayer[label] = 1.0;
  }
}


/// <summary>
/// Train a Recurrent Neural Network model on a test file
/// using the JSON trees of dependency parse
/// </summary>
bool RnnTreeLM::TrainRnnModel() {
  // Reset the log-likelihood to ginourmous value
  double lastValidLogProbability = -1E37;
  double lastValidAccuracy = 0;
  double bestValidLogProbability = -1E37;
  double bestValidAccuracy = 0;
  // Word counter, saved at the end of last training session
  m_wordCounter = m_currentPosTrainFile;
  // Keep track of the initial learning rate
  m_initialLearningRate = m_learningRate;

  // Log file
  string logFilename = m_rnnModelFile + ".log.txt";
  Log("Starting training tree-dependent LM using list of books " +
      m_trainFile + "...\n", logFilename);
  
  bool loopEpochs = true;
  while (loopEpochs) {
    // Reset the log-likelihood of the current iteration
    double trainLogProbability = 0.0;
    // Unique word counter (count only once each word token in a sentence)
    int uniqueWordCounter = 0;
    // Shuffle the order of the books
    m_corpusTrain.ShuffleBooks();

    // Print current epoch and learning rate
    Log("Iter: " + to_string(m_iteration) +
        " Alpha: " + to_string(m_learningRate) + "\n");
    
    // Reset everything, including word history
    ResetAllRnnActivations(m_state);
    
    // Loop over the books
    clock_t start = clock();
    Log(to_string(m_corpusTrain.NumBooks()) + " books to train on\n");
    for (int idxBook = 0; idxBook < m_corpusTrain.NumBooks(); idxBook++) {
      // Read the next book (training file)
      m_corpusTrain.NextBook();
      m_corpusTrain.ReadBook(m_typeOfDepLabels == 1);
      BookUnrolls book = m_corpusTrain.m_currentBook;
      
      // Loop over the sentences in that book
      book.ResetSentence();
      for (int idxSentence = 0; idxSentence < book.NumSentences(); idxSentence++) {
        // Initialize a map of log-likelihoods for each token
        unordered_map<int, double> logProbSentence;
        
        // Loop over the unrolls in each sentence
        book.ResetUnroll();
        int numUnrolls = book.NumUnrolls(idxSentence);
        for (int idxUnroll = 0; idxUnroll < numUnrolls; idxUnroll++) {
          // Reset the state of the neural net before each unroll
          ResetHiddenRnnStateAndWordHistory(m_state);
          // Reset the dependency label features
          // at the beginning of each unroll
          ResetFeatureLabelVector(m_state);
          
          // At the beginning of an unroll,
          // the last word is reset to </s> (end of sentence)
          // and the last label is reset to 0 (root)
          int contextWord = 0;
          int contextLabel = 0;
          
          // Loop over the tokens in the sentence unroll
          bool ok = true;
          while (ok) {

            // Get the current word, discount and label
            int tokenNumber = book.CurrentTokenNumberInSentence();
            int nextContextWord = book.CurrentTokenWordAsContext();
            int targetWord = book.CurrentTokenWordAsTarget();
            double discount = book.CurrentTokenDiscount();
            int targetLabel = book.CurrentTokenLabel();

            // Update the feature matrix with the last dependency label
            if (m_typeOfDepLabels == 2) {
              UpdateFeatureLabelVector(contextLabel, m_state);
            }

            // Run one step of the RNN to predict word
            // from contextWord, contextLabel and the last hidden state
            ForwardPropagateOneStep(contextWord, targetWord, m_state);

            // For perplexity, we do not count OOV words...
            if ((targetWord >= 0) && (targetWord != m_oov)) {
              // Compute the log-probability of the current word
              int outputNodeClass =
              m_vocab.WordIndex2Class(targetWord) + GetVocabularySize();
              double condProbaClass = m_state.OutputLayer[outputNodeClass];
              double condProbaWordGivenClass = m_state.OutputLayer[targetWord];
              double logProbabilityWord =
              log10(condProbaClass * condProbaWordGivenClass);

              // Did we see already that word token (at that position)
              // in the sentence?
              if (logProbSentence.find(tokenNumber) == logProbSentence.end()) {
                // No: store the log-likelihood of that word
                logProbSentence[tokenNumber] = logProbabilityWord;
                // Contribute the log-likelihood to the sentence and corpus
                trainLogProbability += logProbabilityWord;
                uniqueWordCounter++;
              }
              m_wordCounter++;
            }
            
            // Safety check (that log-likelihood does not diverge)
            assert(!(trainLogProbability != trainLogProbability));

            // Shift memory needed for BPTT to next time step
            m_bpttVectors.Shift(contextWord);

            // Discount the learning rate to handle
            // multiple occurrences of the same word
            // in the dependency parse tree
            double alphaBackup = m_learningRate;
            m_learningRate *= discount;
            
            // Back-propagate the error and run one step of
            // stochastic gradient descent (SGD) using optional
            // back-propagation through time (BPTT)
            BackPropagateErrorsThenOneStepGradientDescent(contextWord, targetWord);

            // Undiscount the learning rate
            m_learningRate = alphaBackup;
            
            // Store the current state s(t) at the end of the input layer
            // vector so that it can be used as s(t-1) at the next step
            ForwardPropagateRecurrentConnectionOnly(m_state);
            
            // Rotate the word history by one: the current context word
            // (potentially enriched by dependency label information)
            // will be used at next iteration as input to the RNN
            ForwardPropagateWordHistory(m_state, contextWord, nextContextWord);
            // Update the last label
            contextLabel = targetLabel;

            // Go to the next word
            ok = (book.NextTokenInUnroll() >= 0);
          } // Loop over tokens in the unroll of a sentence
          book.NextUnrollInSentence();

          // Reset the BPTT at every unroll
          m_bpttVectors.Reset();

        } // Loop over unrolls of a sentence
        
        // Verbose
        if (((idxSentence % 1000) == 0) && (idxSentence > 0)) {
          clock_t now = clock();
          double entropy =
          -trainLogProbability/log10((double)2) / uniqueWordCounter;
          double perplexity =
          ExponentiateBase10(-trainLogProbability / (double)uniqueWordCounter);
          Log("Iter," + to_string(m_iteration) +
              ",Alpha," + to_string(m_learningRate) +
              ",Book," + to_string(idxBook) +
              ",TRAINent," + to_string(entropy) +
              ",TRAINppx," + to_string(perplexity) +
              ",words/sec," +
              to_string(1000000 * (m_wordCounter/((double)(now-start)))) + "\n",
              logFilename);
        }
        
        // Reset the table of word token probabilities
        logProbSentence.clear();

        book.NextSentence();
      } // loop over sentences for one epoch

      // Clear memory
      book.Burn();
    } // loop over books for one epoch
    
    // Verbose the iteration
    double trainEntropy = -trainLogProbability/log10((double)2) / uniqueWordCounter;
    double trainPerplexity =
    ExponentiateBase10(-trainLogProbability / (double)uniqueWordCounter);
    clock_t now = clock();
    Log("Iter," + to_string(m_iteration) +
        ",Alpha," + to_string(m_learningRate) +
        ",Book,ALL" +
        ",TRAINent," + to_string(trainEntropy) +
        ",TRAINppx," + to_string(trainPerplexity) +
        ",words/sec," +
        to_string(1000000 * (m_wordCounter/((double)(now-start)))) + "\n",
        logFilename);

    // Validation
    vector<double> sentenceScores;
    double validLogProbability, validPerplexity, validEntropy, validAccuracy;
    TestRnnModel(m_validationFile,
                 m_featureValidationFile,
                 sentenceScores,
                 validLogProbability,
                 validPerplexity,
                 validEntropy,
                 validAccuracy);
    Log("Iter," + to_string(m_iteration) +
        ",Alpha," + to_string(m_learningRate) +
        ",VALIDacc," + to_string(validAccuracy) +
        ",VALIDent," + to_string(validEntropy) +
        ",VALIDppx," + to_string(validPerplexity) +
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
        Log("Saved the best model so far\n", logFilename);
        bestValidAccuracy = validAccuracy;
        bestValidLogProbability = validLogProbability;
      }
    }
  }
  
  return true;
}


/// <summary>
/// Test a Recurrent Neural Network model on a test file
/// </summary>
bool RnnTreeLM::TestRnnModel(const string &testFile,
                             const string &featureFile,
                             vector<double> &sentenceScores,
                             double &logProbability,
                             double &perplexity,
                             double &entropy,
                             double &accuracy) {
  Log("RnnTreeLM::testNet()\n");
  
  // Scores file
  string scoresFilename = m_rnnModelFile + ".scores.";
  size_t sep = testFile.find_last_of("\\/");
  if (sep != string::npos)
    scoresFilename += testFile.substr(sep + 1, testFile.size() - sep - 1);
  scoresFilename += ".iter" + to_string(m_iteration) + ".txt";
  Log("Writing sentence scores to " + scoresFilename + "...\n");

  // We do not use an external file with feature vectors;
  // feature labels are provided in the parse tree itself
  
  // This function does what ResetHiddenRnnStateAndWordHistory does
  // and also resets the features, inputs, outputs and compression layer
  ResetAllRnnActivations(m_state);
  
  // Reset the log-likelihood
  logProbability = 0.0;
  // Reset the unique word token counter
  int uniqueWordCounter = 0;
  int numUnk = 0;
  
  // Since we just set s(1)=0, this will set the state s(t-1) to 0 as well...
  ForwardPropagateRecurrentConnectionOnly(m_state);
  
  // Loop over the books
  if (m_debugMode) { Log("New book\n"); }
  for (int idxBook = 0; idxBook < m_corpusValidTest.NumBooks(); idxBook++) {
    // Read the next book
    m_corpusValidTest.NextBook();
    m_corpusValidTest.ReadBook(m_typeOfDepLabels == 1);
    BookUnrolls book = m_corpusValidTest.m_currentBook;
    
    // Loop over the sentences in the book
    book.ResetSentence();
    if (m_debugMode) { Log("  New sentence\n"); }
    for (int idxSentence = 0; idxSentence < book.NumSentences(); idxSentence++) {
      // Initialize a map of log-likelihoods for each token
      unordered_map<int, double> logProbSentence;
      // Reset the log-likelihood of the sentence
      double sentenceLogProbability = 0.0;
      
      // Loop over the unrolls in each sentence
      book.ResetUnroll();
      int numUnrolls = book.NumUnrolls(idxSentence);
      if (m_debugMode) { Log("    New unroll\n"); }
      for (int idxUnroll = 0; idxUnroll < numUnrolls; idxUnroll++)
      {
        // Reset the state of the neural net before each unroll
        ResetHiddenRnnStateAndWordHistory(m_state);
        // Reset the dependency label features
        // at the beginning of each unroll
        ResetFeatureLabelVector(m_state);
        
        // At the beginning of an unroll,
        // the last word is reset to </s> (end of sentence)
        // and the last label is reset to 0 (root)
        int contextWord = 0;
        int contextLabel = 0;
        
        // Loop over the tokens in the sentence unroll
        bool ok = true;
        while (ok) {
          // Get the current word, discount and label
          int tokenNumber = book.CurrentTokenNumberInSentence();
          int nextContextWord = book.CurrentTokenWordAsContext();
          int targetWord = book.CurrentTokenWordAsTarget();
          int targetLabel = book.CurrentTokenLabel();
          
          if (m_typeOfDepLabels == 2) {
            // Update the feature matrix with the last dependency label
            UpdateFeatureLabelVector(contextLabel, m_state);
          }
          
          // Run one step of the RNN to predict word
          // from contextWord, contextLabel and the last hidden state
          ForwardPropagateOneStep(contextWord, targetWord, m_state);
          
          // For perplexity, we do not count OOV words...
          if ((targetWord >= 0) && (targetWord != m_oov)) {
            // Compute the log-probability of the current word
            int outputNodeClass =
            m_vocab.WordIndex2Class(targetWord) + GetVocabularySize();
            double condProbaClass =
            m_state.OutputLayer[outputNodeClass];
            double condProbaWordGivenClass =
            m_state.OutputLayer[targetWord];
            double logProbabilityWord =
            log10(condProbaClass * condProbaWordGivenClass);
            
            // Did we see already that word token (at that position)
            // in the sentence?
            if (logProbSentence.find(tokenNumber) == logProbSentence.end()) {
              // No: store the log-likelihood of that word
              logProbSentence[tokenNumber] = logProbabilityWord;
              // Contribute the log-likelihood to the sentence and corpus
              logProbability += logProbabilityWord;
              sentenceLogProbability += logProbabilityWord;
              uniqueWordCounter++;
              
              // Verbose
              if (m_debugMode) {
                Log(to_string(tokenNumber) + "\t" +
                    to_string(targetWord) + "\t" +
                    to_string(logProbabilityWord) + "\t" +
                    m_vocab.Word2WordIndex(contextWord) + "\t" +
                    m_corpusValidTest.labelsReverse[contextLabel] + "\t" +
                    m_vocab.Word2WordIndex(targetWord) + "\t" +
                    to_string(m_vocab.WordIndex2Class(targetWord)) + "\t" +
                    to_string(m_vocab.WordIndex2Class(contextWord)) + "\n");
              }
            } else {
              // We have already use the word's log-probability in the score
              // but let's make a safety check
              assert(logProbSentence[tokenNumber] == logProbabilityWord);
              if (m_debugMode) {
                Log(to_string(tokenNumber) + "\t" +
                    to_string(targetWord) + "\t" +
                    to_string(logProbabilityWord) + "\t" +
                    m_vocab.Word2WordIndex(contextWord) + "\t" +
                    m_corpusValidTest.labelsReverse[contextLabel] + "\t" +
                    m_vocab.Word2WordIndex(targetWord) + "(seen)\t" +
                    to_string(m_vocab.WordIndex2Class(targetWord)) + "\t" +
                    to_string(m_vocab.WordIndex2Class(contextWord)) + "\n");
              }
            }
          } else {
            if (m_debugMode) {
              // Out-of-vocabulary words have probability 0 and index -1
              Log(to_string(tokenNumber) + "\t-1\t0\t" +
                  m_vocab.Word2WordIndex(contextWord) + "\t" +
                  m_corpusValidTest.labelsReverse[contextLabel] + "\t" +
                  m_vocab.Word2WordIndex(targetWord) + "\t-1\t-1\n");
            }
            numUnk++;
          }
          
          // Store the current state s(t) at the end of the input layer vector
          // so that it can be used as s(t-1) at the next step
          ForwardPropagateRecurrentConnectionOnly(m_state);
          
          // Rotate the word history by one: the current context word
          // (potentially enriched by dependency label information)
          // will be used at next iteration as input to the RNN
          ForwardPropagateWordHistory(m_state, contextWord, nextContextWord);
          // Update the last label
          contextLabel = targetLabel;
          
          // Go to the next word
          ok = (book.NextTokenInUnroll() >= 0);
        } // Loop over tokens in the unroll of a sentence
        book.NextUnrollInSentence();
      } // Loop over unrolls of a sentence
      
      // Reset the table of word token probabilities
      logProbSentence.clear();
      // Store the log-probability of the sentence
      sentenceScores.push_back(sentenceLogProbability);
      Log(to_string(sentenceLogProbability) + "\n", scoresFilename);

      book.NextSentence();
    } // Loop over sentences
  } // Loop over books
  
  // Log file
  string logFilename = m_rnnModelFile + ".test.log.txt";

  // Return the total logProbability
  Log("Log probability: " + to_string(logProbability) +
      ", number of words " + to_string(uniqueWordCounter) +
      " (" + to_string(numUnk) + " <unk>," +
      " " + to_string(sentenceScores.size()) + " sentences)\n", logFilename);

  // Compute the perplexity and entropy
  perplexity = (uniqueWordCounter == 0) ? 0 :
    ExponentiateBase10(-logProbability / (double)uniqueWordCounter);
  entropy = (uniqueWordCounter == 0) ? 0 :
    -logProbability / log10((double)2) / uniqueWordCounter;
  Log("PPL net (perplexity without OOV): " + to_string(perplexity) + "\n",
      logFilename);

  // Load the labels
  LoadCorrectSentenceLabels(m_fileCorrectSentenceLabels);
  // Compute the accuracy
  accuracy = AccuracyNBestList(sentenceScores, m_correctSentenceLabels);
  Log("Accuracy: " + to_string(accuracy * 100) + "% on " +
      to_string(sentenceScores.size()) + " sentences\n", logFilename);

  return true;
}
