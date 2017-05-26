package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ParamInitializer {

	INDArray init(INDArray view);
	INDArray init(int... shape);
}
