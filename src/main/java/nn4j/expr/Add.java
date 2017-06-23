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

	@Override
	public INDArray doForward() {
		INDArray[] temp = new INDArray[inputs.size()];
		for (int i = 0; i < inputs.size(); i++) {
			temp[i] = inputs.get(i).forward();
		}

		output =Nd4j.getNDArrayFactory().average(temp).muli(inputs.size());

		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		for (int i = 0; i < inputs.size(); i++) {
			INDArray delta = epsilon;
			inputs.get(i).backward(delta);
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}
}
