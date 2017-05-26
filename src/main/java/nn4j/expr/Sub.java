package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Sub extends Expr{

	
	private Expr input1,input2;
	public Sub(Expr input1,Expr input2){
		super(input1,input2);
		this.input1=input1;
		this.input2=input2;
	}
	@Override
	public INDArray doForward() {
		output=input1.forward().sub(input2.forward());
		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input1.backward(epsilon);
		input2.backward(epsilon.neg());
	}
	@Override
	public int[] shape() {
		return input1.shape();
	}

}
