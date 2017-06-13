package nn4j.data;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.expr.Expr;

public class Batch extends Data{
	public Expr[][] batchInputs;//input index, sequence index
	public Expr[] batchOtherInputs;

	public INDArray[] batchMaskings;//input index, sequence length
	public INDArray batchGroundtruth;
}
