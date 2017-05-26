package nn4j.cg;

import java.util.ArrayList;
import java.util.List;

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

	@Override
	public Expr function() {
		List<Expr> pool = new ArrayList<Expr>();
		for (int i = 0; i < units.size() - windowSize; i++) {
			Expr[] concat = new Expr[windowSize];
			for (int j = i; j < i + windowSize; j++) {
				concat[j - i] = units.get(j);
			}
			pool.add(new OuterProduct(new Concat(concat), W));
		}

		switch (poolingType) {
		case Sum:
			return new Add(pool.toArray(new Expr[0]));
		case Avg:
			return new Avg(pool.toArray(new Expr[0]));
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
		return units.get(0).shape();
	}
}
