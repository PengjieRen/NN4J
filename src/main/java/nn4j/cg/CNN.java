package nn4j.cg;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.expr.Add;
import nn4j.expr.Concat;
import nn4j.expr.Dropout;
import nn4j.expr.Expr;
import nn4j.expr.Mask;
import nn4j.expr.Max;
import nn4j.expr.Min;
import nn4j.expr.Parameter;
import nn4j.expr.SMul;

public class CNN extends Vertex {

	private Expr[] units;
	private Parameter W;
	private int windowSize;
	private PoolingType poolingType;
	private boolean training;

	public CNN(Expr[] inputs, Parameter W, int widowSize, PoolingType poolingType, boolean training) {
		this.units = inputs;
		this.W = W;
		this.windowSize = widowSize;
		this.poolingType = poolingType;
		this.training = training;
	}

	public CNN(INDArray maskings, Expr[] inputs, Parameter W, int widowSize, PoolingType poolingType,
			boolean training) {
		super(maskings);
		this.units = inputs;
		this.W = W;
		this.windowSize = widowSize;
		this.poolingType = poolingType;
		this.training = training;
	}

	@Override
	public Expr function() {
		List<Expr> pool = new ArrayList<Expr>();
		for (int i = 0; i < units.length - windowSize; i++) {
			Expr[] concat = new Expr[windowSize];
			INDArray thisPoolMasking = null;
			if (maskings != null) {
				thisPoolMasking = maskings.getColumn(i);
			}
			for (int j = i; j < i + windowSize; j++) {
				concat[j - i] = new Dropout(new Mask(units[j],maskings.getColumn(j)), 0.5f, training);
//				concat[j - i] =new Mask(units[j],maskings.getColumn(j));
				if (maskings != null) {
					thisPoolMasking = thisPoolMasking.mul(maskings.getColumn(j));
				}
			}
			
			if(thisPoolMasking.sumNumber().floatValue()<1)
				break;

			pool.add(new Dense(thisPoolMasking, new Concat(concat), W, Activation.TANH, false, training));
		}

		switch (poolingType) {
		case Sum:
			return new Add(pool.toArray(new Expr[0]));
		case Avg:
			return new SMul(new Add(pool.toArray(new Expr[0])),maskings.sum(1).addi(1).rdiv(1));
//			return new Avg(pool.toArray(new Expr[0]));
		case Max:
			return new Max(pool.toArray(new Expr[0]));
		case Min:
			return new Min(pool.toArray(new Expr[0]));
		default:
			return new Max(pool.toArray(new Expr[0]));
		}
	}

	@Override
	public int[] shape() {
		return units[0].shape();
	}
}
