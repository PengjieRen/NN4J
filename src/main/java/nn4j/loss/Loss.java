package nn4j.loss;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import nn4j.expr.Expr;
import nn4j.utils.NDArrayCache;

public class Loss{

	private Expr prediction;
	private ILossFunction loss;
	private INDArray output;
	private INDArray masking;

	public Loss(Expr prediction,LossFunction loss){
		this.prediction=prediction;
		this.loss=loss.getILossFunction();
	}
	
	public Loss(INDArray masking,Expr prediction,LossFunction loss){
		this.prediction=prediction;
		this.masking=masking;
		this.loss=loss.getILossFunction();
	}
	
	public INDArray forward() {
		output= prediction.forward();
		return output;
	}
	

	public float backward(INDArray gt) {
		IActivation activation=Activation.IDENTITY.getActivationFunction();
		float score=(float)loss.computeScore(gt, output, activation, masking, false);
	    INDArray delta = loss.computeGradient(gt, output, activation, masking);
	    prediction.backward(delta);
	    return score;
	}
	public void clear() {
		if(output!=null)
		{
			NDArrayCache.store(output);
			output=null;
			prediction.clear();
		}
	}
}
