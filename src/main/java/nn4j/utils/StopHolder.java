package nn4j.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.HashSet;
import java.util.Set;

public class StopHolder {

	private Set<String> stopwords;
	
	public StopHolder(File file) {
		stopwords=new HashSet<String>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			while(br.ready()){
				String line=br.readLine().trim();
				stopwords.add(line);
			}
			br.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	public boolean isStopword(String word){
		return stopwords.contains(word.toLowerCase());
	}
}
