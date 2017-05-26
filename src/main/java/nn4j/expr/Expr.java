package nn4j.expr;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.utils.NDArrayCache;

public abstract class Expr {

	protected INDArray output;
	protected List<Expr> inputs=new ArrayList<Expr>();
	
	protected Expr(Expr...exprs){
		for(Expr e : exprs){
			inputs.add(e);
		}
	}
	
	public abstract INDArray doForward();
	
	public INDArray forward(){
		if(output==null)
		output=doForward();
		return output;
	}
	
	public abstract void doBackward(INDArray epsilon);
	
	public void backward(INDArray epsilon){
		doBackward(epsilon);
	}
	
	public void clear() {
		if(output!=null)
		{
			NDArrayCache.store(output);
			output=null;
			for(Expr e : inputs){
				e.clear();
			}
		}
	}
	
	public abstract int[] shape();
}
