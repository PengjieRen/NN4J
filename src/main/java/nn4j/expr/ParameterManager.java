package nn4j.expr;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.expr.Parameter.RegType;
import nn4j.expr.Parameter.Updater;


public class ParameterManager {
	
	private Updater updater;
	public ParameterManager(Updater updater){
		this.updater=updater;
	}

	private List<Parameter> parameters=new ArrayList<Parameter>();
	
	private List<Parameter> tempParameters=new ArrayList<Parameter>();

	public Parameter createParameter(INDArray value,RegType regType,float lambdaReg,boolean updatable,boolean temp){
		Parameter p=new Parameter(value, regType, lambdaReg,updatable,updater);
		if(updatable)
		{
			if(temp){
				tempParameters.add(p);
			}else{
				parameters.add(p);
			}
		}
		return p;
	}
	
	public Parameter createParameter(Parameter param,boolean temp){
		if(param.isUpdatable())
		{
			if(temp){
				tempParameters.add(param);
			}else{
				parameters.add(param);
			}
		}
		return param;
	}
	
	public void update(int iteration) throws Exception{
		if(iteration<=0)throw new Exception("must larger than 0");
		for (int i = 0; i < parameters.size(); i++) {
			parameters.get(i).update(iteration);
		}
		
		for (int i = 0; i < tempParameters.size(); i++) {
			tempParameters.get(i).update(iteration);
		}
		reset();
	}
	
	
	public void reset(){
		for (int i = 0; i < parameters.size(); i++) {
			parameters.get(i).reset();
		}
		for (int i = 0; i < tempParameters.size(); i++) {
			tempParameters.get(i).reset();
		}
		tempParameters.clear();
	}

	public Updater getUpdater() {
		return updater;
	}
	
}
