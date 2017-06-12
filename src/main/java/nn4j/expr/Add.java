package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.utils.NDArrayCache;

/**
 * 
 * @author pengjie ren
 *
 */
public class Add extends Expr {

	public Add(Expr... inputs) {
		super(inputs);
	}

	public Add(INDArray maskings, Expr... inputs) {
		super(maskings, inputs);
	}

	@Override
	public INDArray doForward() {
		INDArray[] temp = new INDArray[inputs.size()];
		for (int i = 0; i < inputs.size(); i++) {
			temp[i] = inputs.get(i).forward();
			if (maskings != null) {
				temp[i] = temp[i].mulColumnVector(maskings.getColumn(i));
			}
		}
		output = NDArrayCache.get(temp[0].shape());

		for (int i = 0; i < inputs.size(); i++) {
			output.addi(temp[i]);
		}

		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		for (int i = 0; i < inputs.size(); i++) {
			INDArray delta = null;
			if (maskings != null) {
				delta = epsilon.mulColumnVector(maskings.getColumn(i));
			} else {
				delta = epsilon;
			}
			inputs.get(i).backward(delta);
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}
}
