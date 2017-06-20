package nn4j.cg;

import nn4j.expr.Expr;
import nn4j.expr.InnerProduct;
import nn4j.expr.OuterProduct;
import nn4j.expr.Parameter;

public class Bilinear extends Vertex{
	
	private Expr in1;
	private Expr in2;
	private Parameter w;
	public Bilinear(Expr in1,Expr in2,Parameter W) {
		this.in1=in1;
		this.in2=in2;
		this.w=W;
	}

	@Override
	public Expr function() {
		return new InnerProduct(new OuterProduct(in1, w), in2);
	}

	@Override
	public int[] shape() {
		return new int[]{in1.shape()[0],1};
	}

}
