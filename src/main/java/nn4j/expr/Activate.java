package nn4j.expr;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.utils.NDArrayCache;

/**
 * 
 * @author pengjie ren
 *
 */
public class Activate extends Expr {

	private Expr input;
	private IActivation activation;
	private boolean training;

	public Activate(Expr input, Activation activation, boolean training) {
		super(input);
		this.input = input;
		this.activation = activation.getActivationFunction();
		this.training = training;
	}

	public Activate(INDArray maskings, Expr input, Activation activation, boolean training) {
		super(maskings, input);
		this.input = input;
		this.activation = activation.getActivationFunction();
		this.training = training;
	}

	private INDArray preout;

	@Override
	public INDArray doForward() {
		preout = input.forward();
		output = NDArrayCache.get(preout.shape());
		output.assign(preout);
		output = activation.getActivation(output, training);
		if (maskings != null) {
			output.muliColumnVector(maskings);
		}
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		INDArray delta = activation.backprop(preout.dup(), epsilon).getFirst();
		if (maskings != null) {
			delta.muliColumnVector(maskings);
		}
		input.backward(delta);
	}

	@Override
	public int[] shape() {
		return input.shape();
	}
}
