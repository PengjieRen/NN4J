package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * 
 * @author pengjie ren
 *
 */
public class Sub extends Expr {

	private Expr input1, input2;
	private INDArray w1, w2;

	public Sub(Expr input1, Expr input2) {
		super(input1, input2);
		this.input1 = input1;
		this.input2 = input2;
	}

	public Sub(INDArray maskings, Expr input1, Expr input2) {
		super(maskings, input1, input2);
		this.input1 = input1;
		this.input2 = input2;
	}

	@Override
	public INDArray doForward() {
		w1 = input1.forward();
		w2 = input2.forward();

		output = w1.sub(w2);
		if (maskings != null) {
			output.muliColumnVector(maskings);
		}
		return w1.sub(w2);
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input1.backward(epsilon.mulColumnVector(maskings));
		input2.backward(epsilon.neg().muliColumnVector(maskings));
	}

	@Override
	public int[] shape() {
		return input1.shape();
	}

}
