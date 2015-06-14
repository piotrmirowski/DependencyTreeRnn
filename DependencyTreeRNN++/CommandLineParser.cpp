// Copyright (c) 2014-2015 Piotr Mirowski
//
// Piotr Mirowski, Andreas Vlachos
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// ACL 2015

#include <iostream>
#include <set>
#include <stdlib.h>
#include "CommandLineParser.h"

using namespace std;


/**
 * Get a command line argument
 */
bool CommandLineParser::Get(string name, int &value) {
  if (args.find(name) == args.end()) {
    cout << name << " must be registered as parameter\n";
    return false;
  }
  CommandLineArgument a = args[name];
  value = atoi(a.m_value.c_str());
  return true;
}


/**
 * Get a command line argument
 */
bool CommandLineParser::Get(string name, double &value) {
  if (args.find(name) == args.end()) {
    cout << name << " must be registered as parameter\n";
    return false;
  }
  CommandLineArgument a = args[name];
  value = atof(a.m_value.c_str());
  return true;
}


/**
 * Get a command line argument
 */
bool CommandLineParser::Get(string name, string &value) {
  if (args.find(name) == args.end()) {
    cout << name << " must be registered as parameter\n";
    return false;
  }
  CommandLineArgument a = args[name];
  value = a.m_value;
  return (!value.empty());
}


/**
 * Get a command line argument
 */
bool CommandLineParser::Get(string name, bool &value) {
  if (args.find(name) == args.end()) {
    cout << name << " must be registered as a parameter\n";
    return false;
  }
  CommandLineArgument a = args[name];
  value = (a.m_value.compare("true") == 0);
  return true;
}


/**
 * Get a command line argument
 */
bool CommandLineParser::Get(string name, long long &value) {
  if (args.find(name) == args.end()) {
    cout << name << " must be registered as a parameter \n";
    return false;
  }
  CommandLineArgument a = args[name];
  value = (long long)atoll(a.m_value.c_str());
  return true;
}


/**
 * Parse the arguments to extract their values and store them in the map
 */
bool CommandLineParser::Parse(char *list[], int llen) {
  if (llen == 1) {
    // Show the arguments
    cout << "Usage: " << list[0]  << "\n";
    for (map<string, CommandLineArgument>::iterator mi = args.begin();
         mi != args.end();
         mi++) {
      if (!mi->second.m_isRequired) {
        cout << "[-" << mi->first << " ("
        << mi->second.m_type << ": "
        << mi->second.m_value << ")]: " << mi->second.m_description << "\n";
      } else {
        cout << "-" << mi->first << " ("
        << mi->second.m_type << "): "
        << mi->second.m_description << "\n";
      }
    }
    return false;
  }
  
  if ((llen % 2) == 0) {
    cout << "Command line must have an even number of elements\n";
    cout << "Check argument structure\n";
    return false;
  }
  
  // List of seen arguments
  set<string> seen;
  for (int i = 1; i < llen; i += 2) {
    if (list[i][0] != '-') {
      cout << "Argument names must begin with -\n";
      cout << "Saw: " << list[i] << endl;
      return false;
    }
    string aname(&list[i][1]);
    if (args.find(aname) == args.end()) {
      cout << "Unknown parameter on command line: " << aname << endl;
      return false;
    }
    args[aname].m_value = string(list[i+1]);
    seen.insert(aname);
  }
  
  // check that the required arguments have been seen
  for (map<string, CommandLineArgument>::iterator mi = args.begin();
       mi != args.end();
       mi++) {
    if (mi->second.m_isRequired && !seen.count(mi->first)) {
      cout << "Required argument " << mi->first << " not set on command line\n";
      return false;
    }
  }
  return true;
}
