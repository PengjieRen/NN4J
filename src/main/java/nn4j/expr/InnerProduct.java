package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public class InnerProduct extends Expr {

	private Expr input1, input2;
	private INDArray w1, w2;

	public InnerProduct(Expr input1, Expr input2) {
		super(input1, input2);
		this.input1 = input1;
		this.input2 = input2;
	}

	@Override
	public INDArray doForward() {
		w1 = input1.forward();
		w2 = input2.forward();

		output = w1.mul(w2);
		output = output.sum(1);
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input1.backward(w2.mulColumnVector(epsilon));
		input2.backward(w1.mulColumnVector(epsilon));
	}

	@Override
	public int[] shape() {
		return new int[] { input1.shape()[0], 1 };
	}

}
