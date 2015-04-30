'''
// Copyright (c) 2014 Anonymized. All rights reserved.
//
// Code submitted as supplementary material for manuscript:
// "Dependency Recurrent Neural Language Models for Sentence Completion"
// Do not redistribute.

Created on Aug 4, 2014

Take a corpus the JSON format obtained from Stanford  and convert it to this

arg1 = input
arg2 = output

corpus = (list of sentences)
sentence = (list of unrolls)
unroll = (list of tokens)
token = (map containing: index in sentence, string, discount, outDep)

outDep is the dependency going from the current token to the next word on the path
the last token on the path (leaf node) has a LEAF outEdge

'''


import networkx
import json
import sys
from collections import Counter
import glob
import os
import os.path

def extractUnrolls(sentenceDAG):
    unrolls = []
    
    # so each unroll is a path from ROOT to the leaves.
    root2leafPaths = []
    # this counts the number of times a node appears in the path
    discountFactors = Counter()
    # traverse all tokens to find the root and the leaves:
    leaves = []
    root = None
    for tokenNo in sentenceDAG.nodes():
        # if a token is a leaf (avoid punctuation which has no incoming ones):
        if sentenceDAG.out_degree(tokenNo) == 0 and sentenceDAG.in_degree(tokenNo) > 0:
            leaves.append(tokenNo)
        if sentenceDAG.in_degree(tokenNo) == 0 and sentenceDAG.out_degree(tokenNo) > 0:
            root = tokenNo
            
    #print "leaves:" + str(leaves)
    #print "root:" + str(root) 
    
    for leaf in leaves:
        # let's get the path from ROOT:
        try:
            path = networkx.shortest_path(sentenceDAG, source=root, target=leaf)
            root2leafPaths.append(path)
            # add the discounts:
            for tok in path:
                discountFactors[tok] += 1
        except networkx.exception.NetworkXNoPath:
            print "path did not exist among tokens " + str(root) + " and " + str(leaf) + " in sentence:"
            print str(sentenceDAG)
    #print root2leafPaths
    #print discountFactors
    
    for path in root2leafPaths:
        unroll = []
        for idx_in_path, tokenNo in enumerate(path):
            #print sentenceDAG[tokenNo]
            word = sentenceDAG.node[tokenNo]['word']
            # the last word has the dummy out edge
            if idx_in_path == len(path)-1:
                outDep = "LEAF"
            else:
                outDep = sentenceDAG[tokenNo][path[idx_in_path+1]]["label"]
            unroll.append([tokenNo, word, discountFactors[tokenNo], outDep]) 
        
        unrolls.append(unroll)
    
    return unrolls

def constructDAG(sentence):
    sentenceDAG = networkx.DiGraph()
    # first put the nodes in the graph
    # fields of interest 0 (tokenNo, starting at 0), 1 (token (lowercase it maybe?), 6 (ancestor), 7 (depLabel to ancestor))
    # add the root
    #sentenceDAG.add_node(0, word="ROOT")
    # add the index of the token in the sentence, remember to start things from 1 as 0 is reserved for root
    for idx, token in enumerate(sentence["tokens"]):
        sentenceDAG.add_node(idx, word=token["word"].lower())
        
    # and now the edges:
    for dependency in sentence["dependencies"]:
        sentenceDAG.add_edge(dependency["head"], dependency["dep"], label=dependency["label"])
    #networkx.draw(sentenceDAG)
    #print sentenceDAG.nodes(data=True)
    #print sentenceDAG.edges(data=True)
    return sentenceDAG

# Create the output path
os.mkdir(sys.argv[2])
threshold = int(sys.argv[3])
tokensOnly = False
# check if we are generating the text for the RNNs
if len(sys.argv) == 5 and sys.argv[4] == "TOKENS":
    tokensOnly = True
    threshold = float("inf")
    
    
tokensKeptCounter = 0
wordTypesKept = []
for filename in glob.glob(sys.argv[1]+ "/*"):
    allSentences = []

    jsonFile = open(filename)
    sentences = json.loads(jsonFile.read())
    jsonFile.close()

    for sentence in sentences:
        sentenceDAG = constructDAG(sentence)
        if (len(sentenceDAG.nodes()) < threshold):
            gutenbergCheck = False
            
            nodes = sentenceDAG.nodes(data=True)

            for node in nodes: 
                if node[1]["word"] == "gutenberg":
                    #print nodes
                    gutenbergCheck = True
            
            if not gutenbergCheck:
                tokensKeptCounter += len(nodes)
                for node in nodes:
                    if node[1]["word"] not in wordTypesKept:
                        wordTypesKept.append( node[1]["word"])
                if tokensOnly:
                    tokens = []
                    for node in nodes:
                        tokens.append(node[1]["word"])
                    allSentences.append(" ".join(tokens))
                else:
                    unrolls = extractUnrolls(sentenceDAG)
                    allSentences.append(unrolls)
    print "unique word types kept=" + str(len(wordTypesKept))    
    if tokensOnly:
        with open(sys.argv[2] + "/" + os.path.basename(filename) + ".tokens.txt", "wb") as out:
            out.write(("\n".join(allSentences)).encode('utf-8') + "\n")
    else:
        with open(sys.argv[2] + "/" + os.path.basename(filename) + ".unrolls.json", "wb") as out:
            json.dump(allSentences, out)

print "tokens kept=" + str(tokensKeptCounter)
print "unique word types kept=" + str(len(wordTypesKept))
