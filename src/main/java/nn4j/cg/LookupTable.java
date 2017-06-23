package nn4j.cg;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;

import nn4j.expr.DefaultParamInitializer;
import nn4j.expr.Expr;
import nn4j.expr.Merge;
import nn4j.expr.Parameter;
import nn4j.expr.Parameter.RegType;
import nn4j.expr.ParameterManager;
import nn4j.expr.WeightInit;

public class LookupTable {

	private INDArray mem;
	private Map<String,Integer> name_row;
	private int dimension;
	private RegType type;
	private float lambdaReg;
	private boolean updatable;
	private ParameterManager pm;
	public LookupTable(ParameterManager pm,int count,int dimension,RegType type,float lambdaReg,boolean updatable) {
		this.pm=pm;
		this.dimension=dimension;
		this.type=type;
		this.lambdaReg=lambdaReg;
		this.updatable=updatable;
		mem=Nd4j.zeros(count,dimension);
		name_row=new HashMap<String, Integer>();
	}
	
	public void add(String name,INDArray value){
		if(!name_row.containsKey(name)){
			int row=name_row.size();
			name_row.put(name, row);
			
			if(value!=null){
				mem.putRow(row, value);
			}else{
				mem.putRow(row, new DefaultParamInitializer(WeightInit.NORMALIZED,new UniformDistribution(-1f/dimension, 1f/dimension)).init(new int[] { 1, dimension }));
			}
		}
	}
	
	public void init(List<String> names){
		for(String name:names){
			add(name, null);
		}
	}
	
	public Expr get(String... names){
		Parameter[] params=new Parameter[names.length];
		for(int i=0;i<names.length;i++){
			params[i]=pm.createParameter(mem.getRow(name_row.get(names[i])), type, lambdaReg, updatable,true);
		}
		
		return new Merge(params);
	}
	
	public void save2File(File file){
		try{
			BufferedWriter bw=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8"));
			for(String key : name_row.keySet()){
				bw.write(key+" "+name_row.get(key));
				bw.newLine();
			}
			bw.write(mem.toString());
			bw.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
}
