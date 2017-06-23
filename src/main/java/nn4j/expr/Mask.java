package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Mask extends Expr{
	
	
	private Expr input;
	private INDArray masking;

	public Mask(Expr input, INDArray masking) {
		super(input);
		this.input = input;
		this.masking=masking;
	}

	@Override
	public INDArray doForward() {
		return input.forward().mulColumnVector(masking);
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input.backward(epsilon.muliColumnVector(masking));
	}

	@Override
	public int[] shape() {
		return input.shape();
	}

}
