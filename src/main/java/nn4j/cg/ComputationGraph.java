package nn4j.cg;

import java.io.File;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

import nn4j.data.Batch;
import nn4j.data.Data;
import nn4j.data.DataLoader;
import nn4j.expr.ParameterManager;
import nn4j.loss.Loss;

public abstract class ComputationGraph {

	protected ParameterManager pm;

	public ComputationGraph(ParameterManager pm) {
		this.pm = pm;
		parameters();
	}

	public abstract void parameters();

	public abstract Loss model(Batch batch, boolean training);
	
	public abstract void test(String run,List<Data> testData,File gt);
	
	public void train(DataLoader trainLoader,List<Data> devData,File devgt,List<Data> testData,File testgt,int iterations) {
		try {
			for (int i = 1; i <= iterations; i++) {
				float loss = 0;
				int batch = 0;
				long begin=System.currentTimeMillis();
				while (trainLoader.hasNext()) {
					batch++;
					Batch data = trainLoader.next();
					Loss model = model(data,true);
					INDArray output=model.forward();
					loss+=model.backward(data.batchGroundtruth);
					pm.update(i);
					if(batch%10==0){
						long end=System.currentTimeMillis();
						System.out.printf("Epoch %s Batch %s Loss %s Time %ss" + System.lineSeparator(), i, batch, loss/batch, (end-begin)/1000);
					}
				}
				
				long end=System.currentTimeMillis();
				System.out.printf("Epoch %s Batch %s Loss %s Time %ss" + System.lineSeparator(), i,batch, loss, (end-begin)/1000);
				if(devData!=null)
				{
					test("dev-"+i,devData,devgt);
				}
				if(testData!=null)
				{
					test("test-"+i,testData,testgt);
				}
				System.out.println("----------------------------------------------");

				trainLoader.reset();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
