package nn4j.expr;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Parameter extends Expr{
	
	public enum RegType{
		None,
		L2
	}

	private INDArray value;
	private List<INDArray> gradients;
	private boolean updatable;
	private RegType regType;
	private float lambdaReg;
	public Parameter(INDArray value,RegType regType,float lambdaReg,boolean updatable){
		this.value=value;
		this.updatable=updatable;
		this.regType=regType;
		this.lambdaReg=lambdaReg;
		this.gradients=new ArrayList<INDArray>();
	}
	
	public RegType regType(){
		return regType;
	}
	
	public float lambdaReg(){
		return lambdaReg;
	}
	
	public boolean updatable(){
		return this.updatable;
	}
	
	public INDArray value(){
		return value;
	}
	
	@Override
	public int[] shape(){
		return value.shape();
	}
	
	public List<INDArray> gradients(){
		return gradients;
	}
	
	@Override
	public INDArray doForward() {
		return value;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		if(updatable){
			gradients.add(epsilon);
		}
	}
}
