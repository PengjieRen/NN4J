package nn4j.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 
 * @author pengjie ren
 *
 */
public class VocabHolder implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2655252883178826426L;

	private Map<String, Integer> vocab2id = new HashMap<String, Integer>();
	private Map<Integer, String> id2vocab = new HashMap<Integer, String>();
	private Map<Integer, Float> id2tf = new HashMap<Integer, Float>();
	private Map<Integer, Float> id2idf = new HashMap<Integer, Float>();
	private Map<Integer, INDArray> id2embed = new HashMap<Integer, INDArray>();
	
	public Map<String, Integer> vocab2id(){
		return vocab2id;
	}
	
	public Map<Integer, String> id2vocab(){
		return id2vocab;
	}

	/**
	 * 
	 * @param embedFile:
	 *            word value1 value2 value3
	 * @param tfFile:
	 *            word tf
	 * @param dfFile:
	 *            word idf
	 * @throws Exception
	 */
	public VocabHolder(String embedFile, String tfFile, String idfFile) {
		if (embedFile != null) {
			readWordEmbedding(new File(embedFile));
		}
		if (tfFile != null) {
			readWordTF(new File(tfFile));
		}
		if (idfFile != null) {
			readWordDF(new File(idfFile));
		}
	}
	public VocabHolder(File embedFile, File tfFile, File idfFile) {
		if (embedFile != null) {
			readWordEmbedding(embedFile);
		}
		if (tfFile != null) {
			readWordTF(tfFile);
		}
		if (idfFile != null) {
			readWordDF(idfFile);
		}
		System.out.println("Vocabulary Size: "+id2vocab().size());
	}
	
	public void addEmbed(String w,INDArray value){
		if(!vocab2id.containsKey(w)){
			int id=vocab2id.size();
			vocab2id.put(w, id);
		    id2vocab.put(id, w);
		    id2embed.put(id, value);
		}
	}
	
	
	public int toID(String word){
		if(vocab2id.containsKey(word))
			return vocab2id.get(word);
		return -1;
	}
	
	public String toVocab(int id){
		if(id2vocab.containsKey(id))
			return id2vocab.get(id);
		return null;
	}
	
	public INDArray toEmbed(String word){
		if(vocab2id.containsKey(word))
		{
			return id2embed.get(vocab2id.get(word));
		}
		return null;
	}
	
	public INDArray toEmbed(int id){
		if(id2vocab.containsKey(id))
		{
			return id2embed.get(id);
		}
		return null;
	}
	
	public INDArray toTFIDFVec(String word){
		INDArray rs=Nd4j.create(new int[]{1, vocab2id.size()},'f');
		if(vocab2id.containsKey(word))
		{
			int id=vocab2id.get(word);
			rs.putScalar(id,id2tf.get(id)*id2idf.get(id));
			return rs;
		}
		return null;
	}
	
	public INDArray toTFIDFVec(int id){
		INDArray rs=Nd4j.create(new int[]{1, id2vocab.size()},'f');
		if(id2vocab.containsKey(id))
		{
			rs.putScalar(id,id2tf.get(id)*id2idf.get(id));
			return rs;
		}
		return null;
	}
	
	public INDArray toOneHotVec(String word){
		INDArray rs=Nd4j.create(new int[]{1, vocab2id.size()},'f');
		if(vocab2id.containsKey(word))
		{
			int id=vocab2id.get(word);
			rs.putScalar(id,1);
			return rs;
		}
		return null;
	}
	
	public INDArray toOneHotVec(int id){
		INDArray rs=Nd4j.create(new int[]{1, id2vocab.size()},'f');
		if(id2vocab.containsKey(id))
		{
			rs.putScalar(id,1);
			return rs;
		}
		return null;
	}

	private void readWordDF(File file) {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(file));

			int count = 0;
			while (br.ready()) {
				if (count++ % 10000 == 0) {
					System.out.printf("[Read] %s [Lines] %s\n", file, count);
				}

				String[] line = br.readLine().split(" ");
				String word = line[0];
				float df = Float.parseFloat(line[1]);
				int id = -1;
				if (!vocab2id.containsKey(word)) {
					id = vocab2id.size();
					vocab2id.put(word, id);
					id2vocab.put(id, word);
				} else {
					System.out.printf("Duplicate word %s\n", word);
					continue;
				}
				id2idf.put(id, df);
			}

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (br != null)
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		}

	}

	private void readWordTF(File file) {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(file));

			int count = 0;
			while (br.ready()) {
				if (count++ % 10000 == 0) {
					System.out.printf("[Read] %s [Lines] %s\n", file, count);
				}

				String[] line = br.readLine().split(" ");
				String word = line[0];
				float tf = Float.parseFloat(line[1]);
				int id = -1;
				if (!vocab2id.containsKey(word)) {
					id = vocab2id.size();
					vocab2id.put(word, id);
					id2vocab.put(id, word);
				} else {
					System.out.printf("Duplicate word %s\n", word);
					continue;
				}
				id2tf.put(id, tf);
			}

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (br != null)
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		}

	}

	private int dimension;
	public int dimension(){
		return dimension;
	}
	private void readWordEmbedding(File file) {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(file));
			String[] temp = br.readLine().split(" ");
			if (temp.length == 2) {
				dimension = Integer.parseInt(temp[1]);
			}

			int count = 0;
			while (br.ready()) {
				if (count++ % 10000 == 0) {
					System.out.printf("[Read] %s [Lines] %s\n", file, count);
				}

				String[] line = br.readLine().split(" ");
				String word = line[0];
				if (line.length != dimension + 1) {
					throw new Exception("Dimension wrong");
				}
				int id = -1;
				if (!vocab2id.containsKey(word)) {
					id = vocab2id.size();
					vocab2id.put(word, id);
					id2vocab.put(id, word);
				} else {
					System.out.printf("Duplicate word %s\n", word);
					continue;
				}
				float[] values = new float[dimension];
				for (int i = 0; i < values.length; i++) {
					values[i] = Float.parseFloat(line[i + 1]);
				}
				id2embed.put(id, Nd4j.create(values, new int[]{1,dimension},'f'));
			}

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (br != null)
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		}
	}
	
	
	

}
