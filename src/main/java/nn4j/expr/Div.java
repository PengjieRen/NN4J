package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Div extends Expr{

	private Expr input1,input2;
	private INDArray w1,w2;
	public Div(Expr input1,Expr input2){
		super(input1,input2);
		this.input1=input1;
		this.input2=input2;
	}
	
	public Div(INDArray maskings,Expr input1,Expr input2){
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
		
		for(int i=0;i<w2.length();i++){
			float v=w2.getFloat(i);
			if(v>0){
				w2.putScalar(i, 1.0f/v);
			}
		}
		
		output=w1.mul(w2);

		return output;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		input1.backward(w2.mul(epsilon));
		input2.backward(w1.mul(epsilon));
	}
	@Override
	public int[] shape() {
		return input1.shape();
	}

}
