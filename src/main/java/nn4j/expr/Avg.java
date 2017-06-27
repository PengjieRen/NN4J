package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 
 * @author pengjie ren
 *
 */
public class Avg extends Expr {

	public Avg(Expr... inputs) {
		super(inputs);
	}

	INDArray sumCount;

	@Override
	public INDArray doForward() {
		inputs.get(0).forward();
		output=Nd4j.zeros(inputs.get(0).shape());
		for (int i = 0; i < inputs.size(); i++) {
			output.addi(inputs.get(i).forward());
		}

		output.divi(inputs.size());
		
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		epsilon=epsilon.div(inputs.size());
		for (int i = 0; i < inputs.size(); i++) {
			inputs.get(i).backward(epsilon);
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}

}
