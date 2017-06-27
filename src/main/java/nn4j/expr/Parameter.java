package nn4j.expr;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaDelta;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.learning.Adam;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.Nesterovs;
import org.nd4j.linalg.learning.NoOpUpdater;

public class Parameter extends Expr {

	public enum RegType {
		None, L2
	}

	private INDArray value;
	private List<INDArray> gradients;
	private boolean updatable;
	private RegType regType;
	private float lambdaReg;
	private GradientUpdater updater;
	
	public Parameter(INDArray value, RegType regType, float lambdaReg, boolean updatable,Updater updater) {
		this.value = value;
		this.updatable = updatable;
		this.regType = regType;
		this.lambdaReg = lambdaReg;
		this.gradients = new ArrayList<INDArray>();
		this.updater=updater(updater);
		
		int[] shape=value.shape();
		int[] nshape=new int[2];
		nshape[0]=shape[0];
		nshape[1]=this.updater.stateSizeForInputSize(shape[1]);
		
		if(nshape[1]>0){
			char order=System.getProperty("ndarray.order").charAt(0);
			this.updater.setStateViewArray(Nd4j.toFlattened(order, Nd4j.zeros(nshape)), shape, order, true);
		}
	}
	
	public boolean isUpdatable() {
		return updatable;
	}

	public void update(int iteration){

		for(INDArray gra : gradients)
		{
			if(regType.equals(RegType.L2)){
				gra.addi(value.mul(lambdaReg));
			}
			gra = updater.getGradient(gra, iteration);
			value.subi(gra);
		}
	}
	
	public void reset(){
		gradients.clear();
	}
	
	
	public enum Updater {
	    SGD
	    ,ADAM
	    ,ADADELTA
	    ,NESTEROVS
	    ,ADAGRAD
	    ,RMSPROP
	    ,NONE
	    ,CUSTOM
	}
	
	private double momentum = 0.9;
	private double adamMeanDecay = 0.9;
	private double adamVarDecay = 0.999;
	private double rho = 0.95;
	private double epsilon = 1e-6;
	private double rmsDecay = 0.95;
	private double learningRate = 1e-3;
	
	public GradientUpdater updater(Updater u) {
		GradientUpdater updater;
		switch (u) {
		case SGD:
			updater = new org.nd4j.linalg.learning.Sgd(learningRate);
			break;
		case ADAM:
			updater = new Adam(learningRate, adamMeanDecay, adamVarDecay);
			break;
		case ADADELTA:
			updater = new AdaDelta(rho, epsilon);
			break;
		case NESTEROVS:
			updater = new Nesterovs(momentum, learningRate);
			break;
		case ADAGRAD:
			updater = new AdaGrad(learningRate, epsilon);
			break;
		case RMSPROP:
			updater = new org.nd4j.linalg.learning.RmsProp(learningRate, rmsDecay);
			break;
		case NONE:
			updater = new NoOpUpdater();
			break;
		case CUSTOM:
			throw new UnsupportedOperationException("Custom updaters: not yet implemented");
		default:
			throw new IllegalArgumentException("Unknown updater: " + u);
		}
		return updater;
	}
	
	public RegType regType() {
		return regType;
	}

	public float lambdaReg() {
		return lambdaReg;
	}

	public boolean updatable() {
		return this.updatable;
	}

	public INDArray value() {
		return value;
	}
	
	public void value(INDArray v){
		value=v;
	}

	@Override
	public int[] shape() {
		return value.shape();
	}

	public List<INDArray> gradients() {
		return gradients;
	}

	@Override
	public INDArray doForward() {
		return value;
	}

	@Override
	public void doBackward(INDArray epsilon) {
		if (updatable) {
			if (gradients.size() == 0)
			{
				gradients.add(epsilon);
			}
			else
			{
				gradients.get(0).addi(epsilon);
			}
		}
	}
}
