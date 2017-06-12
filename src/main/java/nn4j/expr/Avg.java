package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import com.google.common.base.Function;

import nn4j.utils.NDArrayCache;

/**
 * 
 * @author pengjie ren
 *
 */
public class Avg extends Expr {

	public Avg(Expr... inputs) {
		super(inputs);
	}

	public Avg(INDArray maskings, Expr... inputs) {
		super(maskings, inputs);
	}

	INDArray sumCount;

	@Override
	public INDArray doForward() {
		INDArray[] temp = new INDArray[inputs.size()];
		for (int i = 0; i < inputs.size(); i++) {
			temp[i] = inputs.get(i).forward();
			if (maskings != null) {
				temp[i]=temp[i].mulColumnVector(maskings.getColumn(i));
			}
		}
		output = NDArrayCache.get(temp[0].shape());

		for (int i = 0; i < inputs.size(); i++) {
			output.addi(temp[i]);
		}
		if (maskings != null) {
			sumCount = maskings.sum(1);

			BooleanIndexing.applyWhere(sumCount, Conditions.greaterThan(0), new Function<Number, Number>() {
				@Override
				public Number apply(Number arg0) {
					return 1.0f / arg0.floatValue();
				}
			});
			output.muliColumnVector(sumCount);
		} else {
			output.divi(inputs.size());
		}
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		if (maskings != null) {
			epsilon=epsilon.mulColumnVector(sumCount);
		}else{
			epsilon=epsilon.div(inputs.size());
		}
		for (int i = 0; i < inputs.size(); i++) {
			if (maskings != null) {
				inputs.get(i).backward(epsilon.mulColumnVector(maskings.getColumn(i)));
			} else {
				inputs.get(i).backward(epsilon);
			}
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}

}
