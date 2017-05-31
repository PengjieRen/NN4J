package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Sub extends Expr{

	
	private Expr input1,input2;
	private INDArray w1,w2;
	public Sub(Expr input1,Expr input2){
		super(input1,input2);
		this.input1=input1;
		this.input2=input2;
	}
	
	public Sub(INDArray maskings,Expr input1,Expr input2){
		super(maskings,input1,input2);
		this.input1=input1;
		this.input2=input2;
	}
	
	@Override
	public INDArray doForward() {
		w1=input1.forward();
		if(maskings!=null){
			w1.muliColumnVector(maskings.getColumn(0));
		}
		w2=input2.forward();
		if(maskings!=null){
			w2.muliColumnVector(maskings.getColumn(1));
		}
		return w1.sub(w2);
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input1.backward(epsilon.mulColumnVector(maskings.getColumn(0)));
		input2.backward(epsilon.neg().mulColumnVector(maskings.getColumn(1)));
	}
	@Override
	public int[] shape() {
		return input1.shape();
	}

}
