package nn4j.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;

public class VocabHolder {

	private int dimension;
	private int num;
	private String unk="<u>";
	private String pad="<p>";
	private INDArray mem;
	private Map<Integer,INDArray> id2embedding;
	private Map<String,Integer> word2id;
	public VocabHolder(String embedFile){
		loadEmbedding(new File(embedFile));
	}
	
	public VocabHolder(File embedFile){
		loadEmbedding(embedFile);
	}
	
	public int toID(String word){
		if(word2id.containsKey(word))
			return word2id.get(word);
		return word2id.get(unk);
	}
	
	public INDArray toEmbed(String... words){
		int[] tomerge=new int[words.length];
		for(int i=0;i<words.length;i++){
			tomerge[i]=toID(words[i]);
		}
		return mem.getRows(tomerge);
	}
	
	public INDArray toEmbed(String word){
		if(word2id.containsKey(word))
		{
			return id2embedding.get(word2id.get(word));
		}
		return id2embedding.get(word2id.get(unk));
	}
	
	public INDArray toEmbed(int id){
		if(id2embedding.containsKey(id))
		{
			return id2embedding.get(id);
		}
		return null;
	}	
	
	public int dimension(){
		return dimension;
	}
	
	public Set<String> words(){
		return word2id.keySet();
	}
	
	private void loadEmbedding(File file){
		int id = 0;

		BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String[] temp = br.readLine().split(" ");
			if (temp.length == 2) {
				num=Integer.parseInt(temp[0]);
				dimension = Integer.parseInt(temp[1]);
				mem=Nd4j.create(num+2,dimension);
				id2embedding=new HashMap<Integer, INDArray>();
				word2id=new HashMap<String, Integer>();
				
				word2id.put(unk, id);
				mem.putRow(id, Nd4j.rand(new int[]{1, dimension},new UniformDistribution(-0.01f, 0.01f)));
				id2embedding.put(id, mem.getRow(id));
				id++;
				
				word2id.put(pad, id);
				mem.putRow(id, Nd4j.zeros(new int[]{1, dimension}));
				id2embedding.put(id, mem.getRow(id));
				id++;
			}

			while (br.ready()) {
				if (id % 10000 == 0) {
					System.out.printf("[Read] %s [Lines] %s\n", file, id);
				}

				String[] line = br.readLine().split(" ");
				String word = line[0];
				if (line.length != dimension + 1) {
					throw new Exception("Dimension wrong");
				}
				if (!word2id.containsKey(word)) {
					word2id.put(word, id);
					
					for (int i = 1; i < line.length; i++) {
						mem.putScalar(id, i-1, Float.parseFloat(line[i]));
					}
					id2embedding.put(id, mem.getRow(id));
					id++;
				} else {
					System.out.printf("Duplicate word %s\n", word);
					continue;
				}
				
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
