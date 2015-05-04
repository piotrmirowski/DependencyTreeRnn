// Copyright (c) 2014 Anonymized. All rights reserved.
//                                               
// Code submitted as supplementary material for manuscript:
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// Do not redistribute.

#ifndef __DependencyTreeRNN____CommandLineParser__
#define __DependencyTreeRNN____CommandLineParser__

#include <string>
#include <map>

class CommandLineArgument {
public:
    
    /// <summary>
    /// Type of the argument
    /// </summary>
    std::string m_type;
    
    /// <summary>
    /// Description of the argument
    /// </summary>
    std::string m_description;
    
    /// <summary>
    /// Value of the argument
    /// </summary>
    std::string m_value;
    
    /// <summary>
    /// Is the argument required?
    /// </summary>
    bool m_isRequired;
    
    /// <summary>
    /// Constructors
    /// </summary>
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
    /// <summary>
    /// Map between command line argument names and structures containig their values
    /// </summary>
    std::map<std::string, CommandLineArgument> arginfo4;
    
    /// <summary>
    /// Description?
    /// </summary>
    std::string description;
    
    /// <summary>
    /// Register a command line argument
    /// </summary>
    void Register(std::string name,
                  std::string type,
                  std::string desc,
                  std::string defaultVal = "",
                  bool isRequired = false)
    {
        arginfo4[name] = CommandLineArgument(type, desc, defaultVal, isRequired);
    }
    
    /// <summary>
    /// Parse the arguments to extract their values and store them in the map
    /// </summary>
    bool Parse(char *list[], int llen);
    
    /// <summary>
    /// Get a command line argument
    /// </summary>
    bool Get(std::string name, int &value);
    bool Get(std::string name, bool &value);
    bool Get(std::string name, double &value);
    bool Get(std::string name, std::string &value);
    bool Get(std::string name, long long &value);
};

#endif /* defined(__DependencyTreeRNN____CommandLineParser__) */
