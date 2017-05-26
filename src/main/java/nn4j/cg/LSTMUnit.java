package nn4j.cg;

import org.nd4j.linalg.activations.Activation;

import nn4j.expr.Activate;
import nn4j.expr.Add;
import nn4j.expr.Concat;
import nn4j.expr.Expr;
import nn4j.expr.Mul;
import nn4j.expr.Parameter;

/**
 * http://arxiv.org/pdf/1409.2329v5.pdf
 * @author pengjie ren
 *
 */
public class LSTMUnit extends Vertex{

	private Expr h;
	private Expr c;
	private Parameter W_i;
	private Parameter W_f;
	private Parameter W_o;
	private Parameter W_g;
	private Expr in;
	private boolean training;
	public LSTMUnit(Expr prev_h,Expr prev_c,Expr in,Parameter W_i, Parameter W_f, Parameter W_o, Parameter W_g,boolean training) {
		this.h=prev_h;
		this.c=prev_c;
		this.W_i=W_i;
		this.W_f=W_f;
		this.W_o=W_o;
		this.W_g=W_g;
		this.in=in;
		this.training=training;
	}
	@Override
	public Expr function() {
		Expr inConcat=new Concat(h,in);
		Expr i=new Dense(inConcat, W_i, Activation.SIGMOID, true, training);
		Expr f=new Dense(inConcat, W_f, Activation.SIGMOID, true, training);
		Expr o=new Dense(inConcat, W_o, Activation.SIGMOID, true, training);
		Expr g=new Dense(inConcat, W_g, Activation.TANH, true, training);
		c=new Add(new Mul(f,c),new Mul(i,g));
		h=new Mul(o,new Activate(c, Activation.TANH, training));
		return h;
	}

	@Override
	public int[] shape() {
		return new int[]{in.shape()[0],W_o.shape()[1]};
	}

}
