package nn4j.examples;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import nn4j.data.Batch;
import nn4j.data.Data;
import nn4j.data.DataLoader;
import nn4j.expr.Parameter;
import nn4j.expr.Parameter.RegType;
import nn4j.expr.ParameterManager;

public class XORDataLoader extends DataLoader{
	private float[][] inputArr = { { 0.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f } };
	private float[][] outputArr = { { -1f }, { 1f }, { 1f }, { -1f } };
	private boolean hasNext=true;
	
	
	public XORDataLoader(ParameterManager pm) {
		super(pm);
		}
	
	@Override
	public Batch next() {
		hasNext=false;
		Parameter input=new Parameter(Nd4j.create(inputArr),RegType.None,0,false);
		INDArray output = Nd4j.create(outputArr);
		Batch batch=new Batch();
		batch.batchInputs=new Parameter[1][1];
		batch.batchInputs[0]=new Parameter[1];
		batch.batchInputs[0][0]=input;
		batch.batchMaskings=new INDArray[1];
		batch.batchMaskings[0]=Nd4j.create(4, 1).assign(1);
		batch.batchGroundtruth=output;
		return batch;
	}

	@Override
	public boolean hasNext() {
		return hasNext;
	}

	@Override
	public void reset() {
		hasNext=true;
	}

	@Override
	public List<Data> data() {
		List<Data> data=new ArrayList<Data>();
		Parameter input=new Parameter(Nd4j.create(inputArr),RegType.None,0,false);
		INDArray output = Nd4j.create(outputArr);
		Batch batch=new Batch();
		batch.batchInputs=new Parameter[1][1];
		batch.batchInputs[0]=new Parameter[1];
		batch.batchInputs[0][0]=input;
		batch.batchMaskings=new INDArray[1];
		batch.batchMaskings[0]=Nd4j.create(4, 1).assign(1);
		batch.batchGroundtruth=output;
		data.add(batch);
		return data;
	}

}
