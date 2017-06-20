package nn4j.cg;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import nn4j.expr.Activate;
import nn4j.expr.Concat;
import nn4j.expr.Expr;
import nn4j.expr.OuterProduct;
import nn4j.expr.Parameter;
import nn4j.expr.Parameter.RegType;

public class Dense extends Vertex{

	private Parameter W;
	private Activation activation;
	private boolean bias;
	private Expr in;
	private boolean training;
	public Dense(Expr in,Parameter W,Activation activation,boolean bias,boolean training) {
		this.in=in;
		this.W=W;
		this.activation=activation;
		this.bias=bias;
		this.training=training;
	}
	
	public Dense(INDArray maskings,Expr in,Parameter W,Activation activation,boolean bias,boolean training) {
		super(maskings);
		this.in=in;
		this.W=W;
		this.activation=activation;
		this.bias=bias;
		this.training=training;
	}
	
	@Override
	public Expr function() {
		INDArray biasValue=null;
		if(bias){
			biasValue=Nd4j.ones(in.shape()[0],1);
		}else{
			biasValue=Nd4j.zeros(in.shape()[0],1);
		}
		Parameter biasParam=new Parameter(biasValue, RegType.None, 0, false);
		INDArray concatMasking=null;
		if(maskings!=null){
			concatMasking=Nd4j.concat(1, maskings,maskings);
		}
		return new Activate(maskings,new OuterProduct(maskings,new Concat(concatMasking,in,biasParam),W), activation, training);
	}
	@Override
	public int[] shape() {
		return new int[]{in.shape()[0],W.shape()[1]};
	}

}
