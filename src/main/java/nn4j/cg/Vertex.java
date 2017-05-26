package nn4j.cg;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.expr.Expr;

public abstract class Vertex extends Expr{

	private Expr function;
	@Override
	public INDArray doForward() {
		function=function();
		return function.forward();
	}
	
	public abstract Expr function();

	@Override
	public void doBackward(INDArray epsilon) {
		function.backward(epsilon);
	}

	@Override
	public void clear() {
		if(function!=null){
			function.clear();
			function=null;
		}
	}

	
}
