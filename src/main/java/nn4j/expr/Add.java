package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

		output =Nd4j.getNDArrayFactory().average(temp).muli(inputs.size());

		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		for (int i = 0; i < inputs.size(); i++) {
			INDArray delta = epsilon;
			if (maskings != null) {
				delta = delta.mulColumnVector(maskings.getColumn(i));
			} 
			inputs.get(i).backward(delta);
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}
}
