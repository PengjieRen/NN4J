package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

import com.google.common.base.Function;

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
		output = Nd4j.getNDArrayFactory().average(temp);
		
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		if (maskings != null) {
			INDArray sum=maskings.sum(1);
			BooleanIndexing.applyWhere(sum, Conditions.lessThan(0), new Function<Number, Number>() {
				@Override
				public Number apply(Number arg0) {
					return 1.0f/arg0.floatValue();
				}
			});
			epsilon=epsilon.mulColumnVector(sum);
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
