package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Neg extends Expr{

	private Expr input;
	public Neg(Expr input){
		super(input);
		this.input=input;
	}
	@Override
	public INDArray doForward() {
		output=input.forward().neg();
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input.backward(epsilon.neg());
	}
	@Override
	public int[] shape() {
		return input.shape();
	}

}
