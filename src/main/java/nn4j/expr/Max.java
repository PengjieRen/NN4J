package nn4j.expr;

import java.util.ArrayList;
import java.util.List;

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
public class Max extends Expr {

	private INDArray index;

	public Max(Expr... inputs) {
		super(inputs);
	}

	public Max(INDArray maskings, Expr... inputs) {
		super(maskings, inputs);
	}

	@Override
	public INDArray doForward() {

		INDArray maxMaskings = null;
		if (maskings != null) {
			maxMaskings = maskings.dup();
			BooleanIndexing.applyWhere(maxMaskings, Conditions.lessThanOrEqual(0), new Function<Number, Number>() {
				@Override
				public Number apply(Number arg0) {
					return Float.MIN_VALUE;
				}
			});
		}

		List<INDArray> outputs = new ArrayList<INDArray>();

		for (int i = 0; i < inputs.size(); i++) {
			INDArray output_ = inputs.get(i).forward();
			if (maskings != null) {
				output_ = output_.mulColumnVector(maskings.getColumn(i)).addiColumnVector(maxMaskings.getColumn(i));
			}
			outputs.add(output_);
		}
		int[] shape = outputs.get(0).shape();
		INDArray temp = Nd4j.create(outputs, new int[] { outputs.size(), shape[0], shape[1] });

		output = Nd4j.max(temp, 0);
		index = Nd4j.argMax(temp, 0);
		return output;
	}

	private List<INDArray> epsilonNext;

	@Override
	public void doBackward(INDArray epsilon) {
		epsilonNext = new ArrayList<INDArray>();
		int[] shape = index.shape();
		for (int i = 0; i < inputs.size(); i++) {
			epsilonNext.add(Nd4j.zeros(shape));
		}
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				epsilonNext.get(index.getInt(i, j)).putScalar(new int[] { i, j }, epsilon.getFloat(i, j));
			}
		}
		for (int i = 0; i < inputs.size(); i++) {
			inputs.get(i).backward(epsilonNext.get(i));
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}

}
