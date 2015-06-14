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
#include <cstring>
#include <assert.h>
#include <climits>
#include <cmath>
#include <math.h>
#include <algorithm>
#include "Vocabulary.h"


/**
 * Constructor that reads the vocabulary and classes from the model file.
 */
Vocabulary::Vocabulary(FILE *fi, int sizeVocabulary, int numClasses) {
  // Read the vocabulary, stored in text format as following:
  // index_number count word_token class_number
  // There are tabs and spaces separating the 4 columns
  m_vocabularyStorage.resize(sizeVocabulary);
  for (int a = 0; a < sizeVocabulary; a++) {

    // Read the word index and the word count
    int wordIndex;
    int wordCount;
    fscanf(fi, "%d%d", &wordIndex, &wordCount);
    assert(wordIndex == a);
    m_vocabularyStorage[a].cn = wordCount;
    m_vocabularyStorage[a].prob = 0;

    // Read the word token
    char buffer[2048] = {0};
    if (fscanf(fi, "%s", &buffer))
      m_vocabularyStorage[a].word = buffer;
    std::string word = m_vocabularyStorage[a].word;

    // Read the class index
    int classIndex;
    fscanf(fi, "%d", &classIndex);

    // Store the class information
    m_vocabularyStorage[a].classIndex = classIndex;
    m_mapWord2Class[word] = classIndex;

    // Associate the word (string) to the word token number using two maps
    m_mapWord2Index[word] = wordIndex;
    m_mapIndex2Word[wordIndex] = word;
  }

  // Store which words are in which class, using a vector
  // (length number of classes) of vectors (num words in that class)
  m_numClasses = numClasses;
  StoreClassAssociations();

  m_useClassFile = false;
}


/**
 * Save the vocabulary to a model file
 */
void Vocabulary::Save(FILE *fo) {
  // Save the vocabulary, one word per line
  int sizeVocabulary = GetVocabularySize();
  fprintf(fo, "\nVocabulary:\n");
  for (int wordIndex = 0; wordIndex < sizeVocabulary; wordIndex++) {
    int wordCount = m_vocabularyStorage[wordIndex].cn;
    std::string word = m_vocabularyStorage[wordIndex].word;
    int wordClass = m_vocabularyStorage[wordIndex].classIndex;
    fprintf(fo, "%6d\t%10d\t%s\t%d\n",
            wordIndex, wordCount, word.c_str(), wordClass);
  }
}



/**
 * Add a token (word or multi-word entity) to the vocabulary vector
 * and store it in the map from word string to word index
 * and in the map from word index to word string.
 */
int Vocabulary::AddWordToVocabulary(const std::string& word)
{
  int index = SearchWordInVocabulary(word);
  // When a word is unknown, add it to the vocabulary
  if (index == -1) {
    // Initialize the word index, count and probability to 0
    VocabWord w = VocabWord();
    w.word = word;
    w.prob = 0.0;
    w.cn = 1;
    index = static_cast<int>(m_vocabularyStorage.size());
    m_vocabularyStorage.push_back(std::move(w));
    // We need to store the word - index pair in the hash table word -> index
    // but we will rewrite that map later after sorting the vocabulary by frequency
    m_mapWord2Index[word] = index;
    m_mapIndex2Word[index] = word;
  } else {
    // ... otherwise simply increase its count
    m_vocabularyStorage[index].cn++;
  }
  return (index);
}


/**
 * Manually set the word count.
 */
bool Vocabulary::SetWordCount(std::string word, int count) {
  int index = SearchWordInVocabulary(word);
  // When a word is unknown, add it to the vocabulary
  if (index > -1) {
    m_vocabularyStorage[index].cn = count;
    return true;
  } else
    return false;
}


/**
 * Sort the vocabulary by decreasing count of words in the corpus
 * (used for frequency-based word classes, where class 0 contains
 * </s>, class 1 contains {the} or another, most frequent token,
 * class 2 contains a few very frequent tokens, etc...
 */
bool OrderWordCounts(const VocabWord& a, const VocabWord& b) {
  return a.cn > b.cn;
}
void Vocabulary::SortVocabularyByFrequency() {
  // Simply sort the words by frequency, making sure that </s> is first
  int indexEos = SearchWordInVocabulary("</s>");
  int countEos = m_vocabularyStorage[indexEos].cn;
  m_vocabularyStorage[indexEos].cn = INT_MAX;
  std::sort(m_vocabularyStorage.begin(),
            m_vocabularyStorage.end(),
            OrderWordCounts);
  m_vocabularyStorage[indexEos].cn = countEos;

  // Rebuild the the maps of word <-> word index
  m_mapWord2Index.clear();
  m_mapIndex2Word.clear();
  for (int index = 0; index < GetVocabularySize(); index++) {
    std::string word = m_vocabularyStorage[index].word;
    // Add the word to the hash table word -> index
    m_mapWord2Index[word] = index;
    // Add the word to the hash table index -> word
    m_mapIndex2Word[index] = word;
  }
}


/**
 * Return the index of a word in the vocabulary, or -1 if OOV.
 */
int Vocabulary::SearchWordInVocabulary(const std::string& word) const {
  auto i = m_mapWord2Index.find(word);
  if (i == m_mapWord2Index.end()) {
    return -1;
  } else {
    return (i->second);
  }
}


/**
 * Read the classes from a file in the following format:
 * word [TAB] class_index
 * where class index is between 0 and n-1 and there are n classes.
 */
bool Vocabulary::ReadClasses(const std::string &filename)
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
  std::set<std::string> words;
  while (fscanf(fin, "%s%d", w, &clnum) != EOF) {
    if (!strcmp(w, "<s>")) {
      printf("Error: <s> should not be in vocab\n");
      return false;
    }

    m_mapWord2Class[w] = clnum;
    words.insert(w);

    max_class = (clnum > max_class) ? (clnum) : (max_class);
    eos_class = (std::string(w) == "</s>") ? (clnum) : (eos_class);
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
  return true;
}



/**
 * Assign words in vocabulary to classes (for hierarchical softmax).
 */
void Vocabulary::AssignWordsToClasses() {
  int sizeVocabulary = GetVocabularySize();
  if (m_useClassFile) {
    // Custom-specified classes, provided in a file, were used
    // at training time. There is nothing to do at this point,
    // just copy the class index for each word.
    int cnum = -1;
    int last = -1;
    for (int i = 0; i < sizeVocabulary; i++) {
      if (m_vocabularyStorage[i].classIndex != last) {
        last = m_vocabularyStorage[i].classIndex;
        m_vocabularyStorage[i].classIndex = ++cnum;
      } else {
        m_vocabularyStorage[i].classIndex = cnum;
      }
      // Unused
      m_vocabularyStorage[i].prob = 0.0;
    }
  } else {
    // Frequency-based classes (povey-style)
    // Re-assign classes based on the sqrt(word_count / total_word_count)
    // so that the classes contain equal weight of word occurrences.
    int b = 0;
    for (int i = 0; i < sizeVocabulary; i++) {
      b += m_vocabularyStorage[i].cn;
    }
    double dd = 0;
    for (int i = 0; i < sizeVocabulary; i++) {
      dd += sqrt(m_vocabularyStorage[i].cn/ (double)b);
    }
    double df = 0;
    int a = 0;
    for (int i = 0; i < sizeVocabulary; i++) {
      df += sqrt(m_vocabularyStorage[i].cn / (double)b)/dd;
      if (df > 1) {
        df = 1;
      }
      if (df > (a + 1) / (double)m_numClasses) {
        m_vocabularyStorage[i].classIndex = a;
        if (a < m_numClasses - 1) {
          a++;
        }
      } else {
        m_vocabularyStorage[i].classIndex = a;
      }
      // Unused
      m_vocabularyStorage[i].prob = 0.0;
    }
  }

  // Store which words are in which class, using a vector
  // (length number of classes) of vectors (num words in that class)
  StoreClassAssociations();
}


/**
 * Store information on which word is in which class
 */
void Vocabulary::StoreClassAssociations() {
  // Store which words are in which class,
  // using a vector (length number of classes) of vectors (num words in that class)
  m_classWords.resize(m_numClasses);
  for (int i = 0; i < m_numClasses; i++) {
    m_classWords[i].clear();
  }
  for (int i = 0; i < GetVocabularySize(); i++) {
    // Assign each word into its class
    int wordClass = m_vocabularyStorage[i].classIndex;
    m_classWords[wordClass].push_back(i);
  }

  // Check that there is no empty class
  for (int i = 0; i < m_numClasses; i++) {
    assert(!(m_classWords[i].empty()));
  }
}
