package dl4j_RNN;

import org.apache.commons.lang.StringUtils;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

/** A simple DataSetIterator for use in the GravesLSTMWordModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next word in the sequence.<br>
 * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
 * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
 * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
public class WordIterator implements DataSetIterator {
    //Valid characters
    private char[] validCharacters;
    //Maps each character to an index ind the input/output
    private Map<String,Integer> stringToIdxMap = new HashMap<String, Integer>();
    //All words of the input file
    private String[] fileWords;
    //Length of each example/minibatch (number of characters)
    private int exampleLength;
    //Size of each minibatch (number of examples)
    private int miniBatchSize;
    private Random rng;
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param rng Random number generator, for repeatability if required
     * @throws IOException If text file cannot  be loaded
     */
    public WordIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength
                             , Random rng) throws IOException {
        if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
        if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;
        this.rng = rng;

        //Store valid characters in a map for later use in vectorization
//        stringToIdxMap = new HashMap<>();
//        for( int i=0; i<validCharacters.length; i++ ) stringToIdxMap.put(validCharacters[i], i);

        //Load file and convert contents to a char[]
//        boolean newLineValid = stringToIdxMap.containsKey('\n');
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
//        int maxSize = lines.size();	//add lines.size() to account for newline characters at end of each line
//        for( String s : lines ) maxSize += s.length();
        int maxSize = 0;
        for (String s: lines )
        {
            if (! s.isEmpty())
            {
                maxSize += StringUtils.countMatches(s, " ") + 1;
            }
        }
        String[] words = new String[maxSize];
        int currIdx = 0;
        for( String s : lines ){
            String[] thisLine = s.split("\\s+");
            for (String aThisLine : thisLine) {
//                if (!stringToIdxMap.containsKey(aThisLine)) continue;
//                words[currIdx++] = aThisLine;
                if (!stringToIdxMap.containsKey(aThisLine))
                {
                    stringToIdxMap.put(aThisLine,currIdx);
                }
                words[currIdx++] = aThisLine;
            }
        }

        if( currIdx == words.length ){
            fileWords = words;
        } else {
            fileWords = Arrays.copyOfRange(words, 0, currIdx);
        }
        if( exampleLength >= fileWords.length ) throw new IllegalArgumentException("exampleLength="+exampleLength
            +" cannot exceed number of valid characters in file ("+ fileWords.length+")");

        int nRemoved = maxSize - fileWords.length;
        System.out.println("Loaded and converted file: " + fileWords.length + " valid characters of "
            + maxSize + " total characters (" + nRemoved + " removed)");

        initializeOffsets();
    }

    public String convertIndexToCharacter( int idx ){
        return fileWords[idx];
    }

    public int convertStringToIndex(String c ){
        return stringToIdxMap.get(c);
    }

    public String getRandomWord(){
        return fileWords[(int) (rng.nextDouble()*fileWords.length)];
    }

    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    public DataSet next(int num) {
        if( exampleStartOffsets.size() == 0 ) throw new NoSuchElementException();

        int currMinibatchSize = Math.min(num, exampleStartOffsets.size());
        //Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        INDArray input = Nd4j.create(new int[]{currMinibatchSize,fileWords.length,exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize,fileWords.length,exampleLength}, 'f');

        for( int i=0; i<currMinibatchSize; i++ ){
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            int currCharIdx = stringToIdxMap.get(fileWords[startIdx]);	//Current input
            int c=0;
            for( int j=startIdx+1; j<endIdx; j++, c++ ){
                int nextCharIdx = stringToIdxMap.get(fileWords[j]);		//Next character to predict
                input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }

        return new DataSet(input,labels);
    }

    public int totalExamples() {
        return (fileWords.length-1) / miniBatchSize - 2;
    }

    public int inputColumns() {
        return fileWords.length;
    }

    public int totalOutcomes() {
        return fileWords.length;
    }

    public void reset() {
        exampleStartOffsets.clear();
        initializeOffsets();
    }

    private void initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (fileWords.length - 1) / exampleLength - 2;   //-2: for end index, and for partial example
        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets, rng);
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

}
