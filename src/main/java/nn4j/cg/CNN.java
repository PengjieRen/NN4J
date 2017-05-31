package nn4j.cg;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import nn4j.expr.Add;
import nn4j.expr.Avg;
import nn4j.expr.Concat;
import nn4j.expr.Expr;
import nn4j.expr.Max;
import nn4j.expr.Min;
import nn4j.expr.OuterProduct;
import nn4j.expr.Parameter;

public class CNN extends Vertex {

	private List<Expr> units;
	private Parameter W;
	private int windowSize;
	private PoolingType poolingType;

	public CNN(List<Expr> inputs, Parameter W, int widowSize, PoolingType poolingType) {
		this.units = inputs;
		this.W = W;
		this.windowSize = widowSize;
		this.poolingType = poolingType;
	}
	
	public CNN(INDArray maskings,List<Expr> inputs, Parameter W, int widowSize, PoolingType poolingType) {
		super(maskings);
		this.units = inputs;
		this.W = W;
		this.windowSize = widowSize;
		this.poolingType = poolingType;
	}

	@Override
	public Expr function() {
		List<Expr> pool = new ArrayList<Expr>();
		List<INDArray> poolMaskings=null;
		if(maskings!=null){
			poolMaskings=new ArrayList<INDArray>();
		}
		for (int i = 0; i < units.size() - windowSize; i++) {
			Expr[] concat = new Expr[windowSize];
			INDArray thisMasking=null;
			if(maskings!=null){
				thisMasking=maskings.getColumn(i);
			}
			for (int j = i; j < i + windowSize; j++) {
				concat[j - i] = units.get(j);
				if(maskings!=null){
					thisMasking=thisMasking.mul(maskings.getColumn(j));
				}
			}
			
			if(maskings!=null){
				poolMaskings.add(thisMasking);
			}
			pool.add(new OuterProduct(thisMasking,new Concat(concat), W));
		}
		
		INDArray masking=Nd4j.concat(1, poolMaskings.toArray(new INDArray[0]));

		switch (poolingType) {
		case Sum:
			return new Add(masking,pool.toArray(new Expr[0]));
		case Avg:
			return new Avg(masking,pool.toArray(new Expr[0]));
		case Max:
			return new Max(masking,pool.toArray(new Expr[0]));
		case Min:
			return new Min(masking,pool.toArray(new Expr[0]));
		default:
			return new Max(masking,pool.toArray(new Expr[0]));
		}
	}

	@Override
	public int[] shape() {
		return units.get(0).shape();
	}
}
