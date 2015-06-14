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

#ifndef DependencyTreeRNN_Vocabulary_h
#define DependencyTreeRNN_Vocabulary_h

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>


/**
 * Element of vocabulary
 */
struct VocabWord {
  std::string word;
  double prob;
  int cn;
  int classIndex;
};


/**
 * Class storing word in vocabulary, word classes
 * and hash tables to associate them
 */
class Vocabulary {
public:

  /**
   * Constructor.
   */
  Vocabulary(int numClasses)
  : m_numClasses(numClasses), m_useClassFile(false) {
  }

  /**
   * Constructor that reads the vocabulary and classes from the model file.
   */
  Vocabulary(FILE *fi, int sizeVocabulary, int numClasses);

  /**
   * Save the vocabulary to a model file.
   */
  void Save(FILE *fo);

  /**
   * Return the index of a word in the vocabulary, or -1 if OOV.
   */
  int SearchWordInVocabulary(const std::string& word) const;

  /**
   * Add word to the vocabulary.
   */
  int AddWordToVocabulary(const std::string& word);

  /**
   * Sort the words in the vocabulary by frequency.
   */
  void SortVocabularyByFrequency();

  /**
   * Read the classes of words.
   */
  bool ReadClasses(const std::string &filename);

  /**
   * Assign all the words to a class.
   */
  void AssignWordsToClasses();

  /**
   * Return the number of words/entity tokens in the vocabulary.
   */
  int GetVocabularySize() const {
    return static_cast<int>(m_vocabularyStorage.size());
  }

  /**
   * Manually set the word count.
   */
  bool SetWordCount(std::string word, int count);

  /**
   * Return the n-th word in the vocabulary.
   */
  std::string GetNthWord(int word) const {
    return m_vocabularyStorage[word].word;
  }

  /**
   * Return the index of a given word in the vocabulary.
   */
  std::string Word2WordIndex(int word) const {
    return m_vocabularyStorage[word].word;
  }

  /**
   * Return the size of a word class.
   */
  int SizeTargetClass(int targetClass) const {
    return static_cast<int>(m_classWords[targetClass].size());
  }

  /**
   * Return the class index of a word (referenced by an index).
   */
  int WordIndex2Class(int word) const {
    return m_vocabularyStorage[word].classIndex;
  }

  /**
   * Return the n-th word in a word class.
   */
  int GetNthWordInClass(int targetClass, int n) const {
    return static_cast<int>(m_classWords[targetClass][n]);
  }

public:

  // Vocabulary storage
  std::vector<VocabWord> m_vocabularyStorage;

  // Vocabulary representation (word -> index of the word)
  std::unordered_map<std::string, int> m_mapWord2Index;

  // Inverse vocabulary representation (index of the word -> word)
  std::unordered_map<int, std::string> m_mapIndex2Word;

  // Hash table enabling a look-up of the class of a word
  // (word -> word class)
  std::unordered_map<std::string, int> m_mapWord2Class;

  // Information relative to the classes
  std::vector<std::vector<int> > m_classWords;

protected:
  bool m_useClassFile;
  int m_numClasses;

  // Store information on which word is in which class
  void StoreClassAssociations();
}; // class Vocabulary

#endif
