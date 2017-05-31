package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.utils.NDArrayCache;

public class Avg extends Expr{

	public Avg(Expr... inputs){
		super(inputs);
	}
	
	public Avg(INDArray maskings,Expr... inputs) {
		super(maskings,inputs);
	}
	
	
	INDArray sumCount;
	@Override
	public INDArray doForward() {
		INDArray[] temp=new INDArray[inputs.size()];
		for(int i=0;i<inputs.size();i++){
			temp[i]=inputs.get(i).forward();
			if(maskings!=null)
			{
				temp[i].muliColumnVector(maskings.getColumn(i));
			}
		}
		output=NDArrayCache.get(temp[0].shape());

		for(int i=0;i<inputs.size();i++){
			output.addi(temp[i]);
		}
		if(maskings!=null)
		{
			sumCount=maskings.sum(1);
			for(int i=0;i<sumCount.length();i++){
				float c=sumCount.getFloat(i);
				if(c>0){
					sumCount.putScalar(i, 1.0f/c);
				}
			}
			output.muliColumnVector(sumCount);
		}else{
			output.divi(inputs.size());
		}
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		for(int i=0;i<inputs.size();i++){
			if(maskings!=null)
			{
				INDArray delta=epsilon.mulColumnVector(maskings.getColumn(i));
				inputs.get(i).backward(delta.muliColumnVector(sumCount));
			}else{
				inputs.get(i).backward(epsilon.div(inputs.size()));
			}
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}

}
