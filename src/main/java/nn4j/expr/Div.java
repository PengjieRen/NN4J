package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import com.google.common.base.Function;

/**
 * 
 * @author pengjie ren
 *
 */
public class Div extends Expr {

	private Expr input1, input2;
	private INDArray w1, w2;

	public Div(Expr input1, Expr input2) {
		super(input1, input2);
		this.input1 = input1;
		this.input2 = input2;
	}

	@Override
	public INDArray doForward() {
		w1 = input1.forward();
		w2 = input2.forward();

		BooleanIndexing.applyWhere(w2, Conditions.greaterThan(0), new Function<Number, Number>() {
			@Override
			public Number apply(Number arg0) {
				return 1.0f / arg0.floatValue();
			}
		});

		output = w1.mul(w2);

		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input1.backward(w2.mul(epsilon));
		input2.backward(w1.mul(epsilon));
	}

	@Override
	public int[] shape() {
		return input1.shape();
	}

}
