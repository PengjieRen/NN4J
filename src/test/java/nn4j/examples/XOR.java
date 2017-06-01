package nn4j.examples;

import java.io.File;
import java.util.List;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import nn4j.cg.ComputationGraph;
import nn4j.cg.Dense;
import nn4j.data.Batch;
import nn4j.data.Data;
import nn4j.data.DataLoader;
import nn4j.expr.DefaultParamInitializer;
import nn4j.expr.Expr;
import nn4j.expr.ParamInitializer;
import nn4j.expr.Parameter;
import nn4j.expr.Parameter.RegType;
import nn4j.expr.ParameterManager;
import nn4j.expr.ParameterManager.Updater;
import nn4j.expr.WeightInit;
import nn4j.loss.Loss;

public class XOR extends ComputationGraph{

	private Parameter w1;
	private Parameter w2;
	private Parameter w3;

	public XOR(ParameterManager pm) {
		super(pm);
	}
	
	@Override
	public void parameters() {
		ParamInitializer init=new DefaultParamInitializer(WeightInit.DISTRIBUTION,new UniformDistribution(-0.1, 0.1));
		w1 = pm.createParameter(init.init(new int[] { 3, 100 }),RegType.None,0, true);
		w2 = pm.createParameter(init.init(new int[] { 101, 100 }),RegType.None,0, true);
		w3 = pm.createParameter(init.init(new int[] { 101, 1 }),RegType.None,0, true);
	}

	@Override
	public Loss model(Parameter[][] inputs, boolean training) {
		Expr v1 = new Dense(inputs[0][0],w1,Activation.SIGMOID,false,training);
		Expr v2 = new Dense(v1,w2,Activation.SIGMOID,false,training);
		Expr v3 = new Dense(v2,w3,Activation.SIGMOID,false,training);
		Loss loss = new Loss(v3, LossFunction.MSE);
		return loss;
	}

	@Override
	public void test(String run,List<Data> testData,File gt) {
		for(Data data : testData){
			Batch ins=(Batch)data;
			Loss loss=model(ins.batchInputs,false);
			INDArray output=loss.forward();
			System.out.println(ins.batchInputs[0][0].value()+"=>"+output);
			loss.clear();
		}
	}

	public static void main(String[] args) throws Exception {
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
		
		DataLoader loader=new XORDataLoader();
		ParameterManager pm = new ParameterManager(Updater.ADADELTA);
		XOR model = new XOR(pm);
		
		model.train(loader,loader.data(),null,null,null,1000000);
	}

	

}
