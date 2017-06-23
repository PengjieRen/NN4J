package nn4j.cg;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.expr.Add;
import nn4j.expr.Concat;
import nn4j.expr.Expr;
import nn4j.expr.Mul;
import nn4j.expr.Neg;
import nn4j.expr.Parameter;

/**
 * https://arxiv.org/pdf/1406.1078v3.pdf
 * @author pengjie ren
 *
 */
public class GRUUnit extends Vertex{

	private Expr h;
	private Parameter W_z;
	private Parameter W_r;
	private Parameter W;
	private Expr in;
	private boolean training;
	public GRUUnit(Expr prev_h,Expr in,Parameter W_z, Parameter W_r, Parameter W, boolean training) {
		this.h=prev_h;
		this.W_z=W_z;
		this.W_r=W_r;
		this.W=W;
		this.in=in;
		this.training=training;
	}
	
	public GRUUnit(INDArray maskings,Expr prev_h,Expr in,Parameter W_z, Parameter W_r, Parameter W, boolean training) {
		super(maskings);
		this.h=prev_h;
		this.W_z=W_z;
		this.W_r=W_r;
		this.W=W;
		this.in=in;
		this.training=training;
	}
	
	@Override
	public Expr function() {
		Expr inConcat=new Concat(h,in);
		Expr z=new Dense(maskings,inConcat, W_z, Activation.SIGMOID, true, training);
		Expr r=new Dense(maskings,inConcat, W_r, Activation.SIGMOID, true, training);
		Expr h_=new Dense(maskings,new Concat(new Mul(r,h),in), W, Activation.TANH, true, training);
		h=new Add(h,new Neg(new Mul(z,h)),new Mul(z,h_));
		return h;
	}

	@Override
	public int[] shape() {
		return new int[]{in.shape()[0],W_z.shape()[1]};
	}

}
