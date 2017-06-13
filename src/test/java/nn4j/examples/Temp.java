package nn4j.examples;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Temp {

	public static void main(String[] args) {
		INDArray input1=Nd4j.rand(new int[]{2,3});
		INDArray input2=Nd4j.rand(new int[]{2,3});
		System.out.println(input1);
		System.out.println(input2);
		
		INDArray input3=Nd4j.concat(1, input1,input2);
		System.out.println(input3);
		input3.putScalar(1, 111);
		System.out.println(input3);
		System.out.println(input1);
		
//		IActivation activation=Activation.SOFTMAX.getActivationFunction();
//		INDArray input=Nd4j.rand(new int[]{2,3});
//		
//		INDArray preout=input;
//		preout=preout.mul(preout);
//		INDArray l2=preout.sum(1);
//		l2=l2.broadcast(preout.shape());
//		INDArray output=preout.div(l2);
//		System.out.println(output);
//		
//		INDArray backLoss=l2.sub(1).muli(2).muli(output).divi(l2.mul(l2));
//		System.out.println(backLoss);
		
//		INDArray norm2=input1.norm2(1);
//		System.out.println(input1);
//		System.out.println(norm2);
//		System.out.println(norm2.broadcast(input1.shape()));
		
////		System.out.println(input1);
//		INDArray input2=Nd4j.rand(new int[]{2,3});
////		System.out.println(input2);
//		List<INDArray> arrays=new ArrayList<INDArray>();
//		arrays.add(input1);
//		arrays.add(input2);
////		System.out.println(Nd4j.create(arrays, new int[]{2,2,3},'f'));
//		System.out.println(Nd4j.create(arrays, new int[]{2,2,3},'c'));
//		System.out.println(Nd4j.max(Nd4j.create(arrays, new int[]{2,2,3},'c'),0));
////		System.out.println(activation.getActivation(input1, false));
	}

}
