package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 
 * @author pengjie ren
 *
 */
public class Dropout extends Expr {

	private Expr input;
	private boolean training;
	private float rejectProb;

	public Dropout(Expr input, float acceptProb, boolean training) {
		super(input);
		this.input = input;
		this.rejectProb = 1.0f - acceptProb;
		this.training = training;
	}

	private INDArray acceptArray;

	@Override
	public INDArray doForward() {
		INDArray preout = input.forward();
		acceptArray = Nd4j.zeros(preout.shape());
		if (training) {
			Nd4j.getExecutioner().exec(new BernoulliDistribution(acceptArray, rejectProb));
		} else {
			acceptArray.assign(1.0-rejectProb);
		}
		output = preout.mul(acceptArray);
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		INDArray delta = epsilon.mul(acceptArray);
		input.backward(delta);
	}

	@Override
	public int[] shape() {
		return input.shape();
	}

}
