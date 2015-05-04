import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.*;

import java.io.*;
import java.util.*;

import com.google.gson.*;



public class Text2Parsed2JSON {
	
	public class MyToken{
		public String word;
		public String lemma;
		public String pos;
		public String ner;
		
		public MyToken(){};
		
	}
	
	public class MyDependency{
		public int head;
		public int dep;
		public String label;
		
		public MyDependency(){};
	}
	
	public class MySentence{
		
		public ArrayList<MyToken> tokens;
		public ArrayList<MyDependency> dependencies;
		
		public MySentence(){
			tokens = new ArrayList<MyToken>();
			dependencies = new ArrayList<MyDependency>();
		}
		
	}

	// this holds the main pipeline for the processing 
	private StanfordCoreNLP mainPipeline;
	
    public static String readTextFromFile(File textFileName) throws IOException {
    	BufferedReader textFile = new BufferedReader(new FileReader(textFileName));
    	String line;
    	StringBuffer result = new StringBuffer();
    	while ((line = textFile.readLine() ) != null){
    		// added the new line back
    		result.append(line + "\n");
    	}
    	textFile.close();
    	return result.toString();
    }
    
    // dummy function that returns the same text that was passed as input.
    // to be over-ridden to do more interesting things. might need to add to the initialization.
    private String filterText(String text){
    	return text;
    }
    
    public Text2Parsed2JSON(){
		// Initialize the parser:
		Properties parser_props = new Properties();
		parser_props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse");
		// I assume that longer sentences are unlikely to be useful.
		parser_props.put("parse.maxlen", 80);
		//parser_props.put("tokenize.whitespace", "true");
		//parser_props.put("ssplit.isOneSentence", "true");
		mainPipeline = new StanfordCoreNLP(parser_props);		
    	
    }
    
    // this takes text, runs the main processor and returns the Stanford annotations
    // for the sentences kept 
    public Annotation processText2Annotations(String text){
    	// filter the text
    	String filteredText = filterText(text); 
    	// create an empty Annotation just with the given text
        Annotation annotatedText = new Annotation(filteredText);
                
        mainPipeline.annotate(annotatedText);

        return annotatedText;
    }


    public String processAnnotations2JSON(Annotation annotatedText){
    	
    	// initialize the sentences array
    	ArrayList<MySentence> mySentences = new ArrayList<MySentence>();
    	
    	// get the sentences 
        List<CoreMap> sentences = annotatedText.get(SentencesAnnotation.class);
        
        for(CoreMap sentence: sentences) {
        	MySentence newSentence = new MySentence();
          // traversing the words in the current sentence
          // a CoreLabel is a CoreMap with additional token-specific methods
          for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
        	  MyToken newToken = new MyToken();
            // this is the text of the token
            String word = token.get(TextAnnotation.class);
            // this is the POS tag of the token
            String pos = token.get(PartOfSpeechAnnotation.class);
            // this is the NER label of the token
            String ne = token.get(NamedEntityTagAnnotation.class);
            // this is the lemma
            String lemma = token.get(CoreAnnotations.LemmaAnnotation.class);
            newToken.lemma = lemma;
            newToken.pos = pos;
            newToken.ner = ne;
            newToken.word = word;
            
            newSentence.tokens.add(newToken);
          }


          // this is the Stanford dependency graph of the current sentence
          // If a tree with all the tokens is required, use BasicDependenciesAnnotation
          // But the one that are commonly the best for IE are CollapsedCCProcessedDependenciesAnnotation (careful, they are not even DAGs)
          SemanticGraph dependencies = sentence.get(BasicDependenciesAnnotation.class);
         //System.out.print(dependencies.toString("plain"));
          
          //Set<SemanticGraphEdge> allEdges = dependencies.getEdgeSet();
          
          for (SemanticGraphEdge edge: dependencies.edgeIterable()){
        	  MyDependency dep = new MyDependency();
        	  // remember to subtract one so that the first word starts at 0
        	  dep.head = edge.getGovernor().index() - 1;
        	  dep.dep = edge.getDependent().index() - 1;
        	  dep.label = edge.getRelation().toString();
        	  
        	  newSentence.dependencies.add(dep);
          }
          
          mySentences.add(newSentence);  
        }
        
        
    	Gson gson = new Gson();
    	
    	
    	return gson.toJson(mySentences);
    }

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// initialize
		Text2Parsed2JSON processor = new Text2Parsed2JSON();
		
		// get the directory with the text files
		File extractsDirectory = new File(args[0]);

		// get the output directory
		File outputDirectory = new File(args[1]);
		outputDirectory.mkdir();

		// get a list of files:
		File[] textFileNames = extractsDirectory.listFiles();
		System.out.println("Files to process:" +  textFileNames.length);

		// For each text file:
		for (int i = 0; i < textFileNames.length; i++){
			
			// First get the filename
			//String filename = textFileNames[i].getName();
			System.out.println(textFileNames[i]);
			// Read in the text
			String text;
			try {
				text = readTextFromFile(textFileNames[i]);
				// process
				Annotation annotatedText = processor.processText2Annotations(text);
				String JSONsentences = processor.processAnnotations2JSON(annotatedText);
				//System.out.println(JSONsentences);
				    
				// Create the file for the output
				File JSONFile = new File(outputDirectory, textFileNames[i].getName() + ".json");
				//System.out.println(JSONFile.getPath());
				//System.out.println(JSONFile.getName());
				BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(JSONFile), "utf-8"));
				out.write(JSONsentences);
				out.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block 
				e.printStackTrace();
			}
			    

		
		}
	}

}
