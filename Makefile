CC = g++

BLASFLAGS = -I/opt/local/include
CPPFLAGS = -Wall -O3 -std=c++0x
OPTIMFLAGS = -funroll-loops -ffast-math
CXXFLAGS = -lm -lblas -g $(CPPFLAGS) $(OPTIMFLAGS) $(BLASFLAGS)

LDFLAGS = -lblas

BLASINCLUDE = /opt/local/include/cblas.h
SRCDIR = DependencyTreeRNN++
INCLUDES = $(BLASINCLUDE) $(SRCDIR)/*.h

OBJDIR = build

OBJ =	$(OBJDIR)/ReadJson.o \
	$(OBJDIR)/CorpusUnrollsReader.o \
	$(OBJDIR)/CommandLineParser.o \
	$(OBJDIR)/Vocabulary.o \
	$(OBJDIR)/RnnWeights.o \
	$(OBJDIR)/RnnLib.o \
	$(OBJDIR)/RnnTraining.o \
	$(OBJDIR)/RnnDependencyTreeLib.o \
	$(OBJDIR)/main.o

all: $(OBJ) RnnDependencyTree

$(OBJDIR)/ReadJson.o: $(SRCDIR)/ReadJson.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/CorpusUnrollsReader.o: $(SRCDIR)/CorpusUnrollsReader.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/CommandLineParser.o: $(SRCDIR)/CommandLineParser.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/Vocabulary.o: $(SRCDIR)/Vocabulary.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/RnnWeights.o: $(SRCDIR)/RnnWeights.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/RnnLib.o: $(SRCDIR)/RnnLib.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/RnnTraining.o: $(SRCDIR)/RnnTraining.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/RnnDependencyTreeLib.o: $(SRCDIR)/RnnDependencyTreeLib.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(INCLUDES)
	$(CC) $(CXXFLAGS) -c -o $@ $<

RnnDependencyTree: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(OBJDIR)/*.o
