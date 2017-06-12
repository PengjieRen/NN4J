package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * 
 * @author pengjie ren
 *
 */
public class Concat extends Expr {

	private int[] lengths;
	private int length;

	public Concat(Expr... inputs) {
		super(inputs);
		for (int i = 0; i < inputs.length; i++) {
			length += inputs[i].shape()[1];
		}
	}

	public Concat(INDArray maskings, Expr... inputs) {
		super(maskings, inputs);
		for (int i = 0; i < inputs.length; i++) {
			length += inputs[i].shape()[1];
		}
	}

	@Override
	public INDArray doForward() {
		INDArray[] outputs = new INDArray[inputs.size()];
		lengths = new int[inputs.size()];
		for (int i = 0; i < inputs.size(); i++) {
			outputs[i] = inputs.get(i).forward();
			if (maskings != null) {
				outputs[i] = outputs[i].mulColumnVector(maskings.getColumn(i));
			}else{
				outputs[i] =outputs[i].dup();
			}
			lengths[i] = outputs[i].shape()[1];
		}
		output = Nd4j.concat(1, outputs);
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		int st = 0;
		for (int i = 0; i < inputs.size(); i++) {
			INDArray delta = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(st, st + lengths[i]));
			st += lengths[i];
			if (maskings != null) {
				delta = delta.mulColumnVector(maskings.getColumn(i));
			}
			inputs.get(i).backward(delta);
		}
	}

	@Override
	public int[] shape() {
		return new int[] { inputs.get(0).shape()[0], length };
	}

}
