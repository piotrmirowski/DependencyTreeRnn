// Copyright (c) 2014 Piotr Mirowski. All rights reserved.
//                    piotr.mirowski@computer.org
// Based on code by Davide Di Gennaro (dadigenn@microsoft.com)

#ifndef DependencyTreeRNN___CorpusWordReader_h
#define DependencyTreeRNN___CorpusWordReader_h

#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>


inline bool isSpace(char c) { return isspace(c); };
inline bool notIsSpace(char c) { return !isspace(c); };


/// <summary>
/// Simple class to read words, one by one, from a file.
/// When the end of a line is reached, it returns "</s>"
/// </summary>
class WordReader {
protected:
  std::ifstream m_file;
  std::string m_line;

public:
    
  WordReader(const std::string &filename)
  : m_file(filename) {
  }
    

  std::string pop_first_word(std::string &s) {
    const auto p1 = std::find_if(s.begin(), s.end(), notIsSpace);
    const auto p2 = std::find_if(p1, s.end(), isSpace);
    const std::string word(p1, p2);
    s.erase(0, std::find_if(p2, s.end(), notIsSpace) - s.begin());
    return word;
  }
    
    
  std::string get_next() {
    std::string result;
    if (m_line.empty()) {
      if (std::getline(m_file, m_line)) {
        m_line += " </s>";
      } else {
        return result;
      }
    }
    result = pop_first_word(m_line);
    return result;
  }
};

#endif
