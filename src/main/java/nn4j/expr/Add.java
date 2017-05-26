package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.utils.NDArrayCache;

public class Add extends Expr{

	public Add(Expr... inputs){
		super(inputs);
	}
	
	@Override
	public INDArray doForward() {
		INDArray[] temp=new INDArray[inputs.size()];
		for(int i=0;i<inputs.size();i++){
			temp[i]=inputs.get(i).forward();
		}
		output=NDArrayCache.get(temp[0].shape());

		for(int i=0;i<inputs.size();i++){
			output.addi(temp[i]);
		}

		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		for(int i=0;i<inputs.size();i++){
			inputs.get(i).backward(epsilon);
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}
}
