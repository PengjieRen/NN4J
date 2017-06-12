package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 
 * @author pengjie ren
 *
 */
public class Neg extends Expr {

	private Expr input;

	public Neg(Expr input) {
		super(input);
		this.input = input;
	}

	public Neg(INDArray maskings, Expr input) {
		super(maskings, input);
		this.input = input;
	}

	@Override
	public INDArray doForward() {
		output = input.forward().neg();
		if(maskings!=null)
		{
			output=output.muliColumnVector(maskings);
		}
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input.backward(epsilon.neg().muliColumnVector(maskings));
	}

	@Override
	public int[] shape() {
		return input.shape();
	}

}
