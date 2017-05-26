package nn4j.data;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.expr.Parameter;

public class Batch extends Data{
	public Parameter[][] batchInputs;//input index, sequence index
	public int[][] batchMaskings;//input index, sequence length
	public INDArray batchGroundtruth;
}
