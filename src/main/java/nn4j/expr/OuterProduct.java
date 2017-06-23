package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public class OuterProduct extends Expr {

	private Expr input1, input2;
	private INDArray w1, w2;

	public OuterProduct(Expr input1, Parameter input2) {
		super(input1, input2);
		this.input1 = input1;
		this.input2 = input2;
	}

	@Override
	public INDArray doForward() {
		w1 = input1.forward();
		w2 = input2.forward();

		output = w1.mmul(w2);

		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input2.backward(w1.transpose().mmul(epsilon));
		input1.backward(epsilon.mmul(w2.transpose()));
	}

	@Override
	public int[] shape() {
		return new int[] { input1.shape()[0], input2.shape()[1] };
	}
}
