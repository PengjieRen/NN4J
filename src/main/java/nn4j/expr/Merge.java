package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Merge extends Expr{

	private int[] lengths;
	private int length;
	public Merge(Expr... inputs){
		super(inputs);
		for (int i = 0; i < inputs.length; i++) {
			length+=inputs[i].shape()[0];
		}
	}
	
	@Override
	public INDArray doForward() {
		INDArray[] outputs = new INDArray[inputs.size()];
		lengths=new int[inputs.size()];
		for (int i = 0; i < inputs.size(); i++) {
			outputs[i] = inputs.get(i).forward();
			lengths[i]=outputs[i].shape()[0];
		}
		output = Nd4j.concat(0, outputs);
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		int st = 0;
		for (int i = 0; i < inputs.size(); i++) {
			INDArray backLoss = epsilon.get(NDArrayIndex.interval(st, st + lengths[i]),NDArrayIndex.all());
			st += lengths[i];
			inputs.get(i).backward(backLoss);
		}
	}

	@Override
	public int[] shape() {
		return new int[]{length,inputs.get(0).shape()[1]};
	}

}
