package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public class SMul extends Expr {

	private Expr input1;
	private INDArray input2;

	public SMul(Expr input1, INDArray input2) {
		super(input1);
		this.input1 = input1;
		this.input2 = input2;
	}

	
	@Override
	public INDArray doForward() {
		return input1.forward().muliColumnVector(input2);
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input1.backward(epsilon.muliColumnVector(input2));
	}

	@Override
	public int[] shape() {
		return input1.shape();
	}

}
