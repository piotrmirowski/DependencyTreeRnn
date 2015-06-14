// Copyright (c) 2014-2015 Piotr Mirowski
//
// Piotr Mirowski, Andreas Vlachos
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// ACL 2015

#ifndef __DependencyTreeRNN____CommandLineParser__
#define __DependencyTreeRNN____CommandLineParser__

#include <string>
#include <map>

class CommandLineArgument {
public:
    
  /**
   * Type of the argument
   */
  std::string m_type;

  /**
   * Description of the argument
   */
  std::string m_description;
    
  /**
   * Value of the argument
   */
  std::string m_value;
  
  /**
   * Is the argument required?
   */
  bool m_isRequired;
    
  /**
   * Constructors
   */
  CommandLineArgument(std::string t,
                      std::string desc,
                      std::string d,
                      bool r)
  : m_type(t), m_description(desc), m_value(d), m_isRequired(r) {
  }
  CommandLineArgument() {
    m_type = "UNDEFINED";
  }
};


class CommandLineParser {
public:
  /**
   * Map between command line argument names and structures containig their values
   */
  std::map<std::string, CommandLineArgument> args;

  /**
   * Register a command line argument
   */
  void Register(std::string name,
                std::string type,
                std::string desc,
                std::string defaultVal = "",
                bool isRequired = false) {
    args[name] = CommandLineArgument(type, desc, defaultVal, isRequired);
  }
  
  /**
   * Parse the arguments to extract their values and store them in the map
   */
  bool Parse(char *list[], int llen);
  
  /**
   * Get a command line argument
   */
  bool Get(std::string name, int &value);
  bool Get(std::string name, bool &value);
  bool Get(std::string name, double &value);
  bool Get(std::string name, std::string &value);
  bool Get(std::string name, long long &value);
};

#endif /* defined(__DependencyTreeRNN____CommandLineParser__) */
