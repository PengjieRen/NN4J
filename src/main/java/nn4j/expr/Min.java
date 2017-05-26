package nn4j.expr;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import nn4j.utils.NDArrayCache;

public class Min extends Expr{

	private INDArray index;
	public Min(Expr... inputs){
		super(inputs);
	}
	
	@Override
	public INDArray doForward() {
		List<INDArray> outputs = new ArrayList<INDArray>();

		for (int i = 0; i < inputs.size(); i++) {
			outputs.add(inputs.get(i).forward());
		}
		int[] shape=outputs.get(0).shape();
		INDArray temp=NDArrayCache.create(outputs, new int[]{outputs.size(),shape[0],shape[1]});

		output=Nd4j.min(temp,0);
		index=Nd4j.argMax(temp.mul(-1), 0);
		return output;
	}

	private List<INDArray> epsilonNext;
	@Override
	public void doBackward(INDArray epsilon) {
		epsilonNext = new ArrayList<INDArray>();
		int[] shape=index.shape();
		for (int i = 0; i < inputs.size(); i++) {
			epsilonNext.add(NDArrayCache.get(shape));
		}
		for(int i=0;i<shape[0];i++){
			for(int j=0;j<shape[1];j++){
				epsilonNext.get(index.getInt(i,j)).putScalar(new int[]{i, j},epsilon.getFloat(i,j));
			}
		}
		for (int i = 0; i < inputs.size(); i++) {
			inputs.get(i).backward(epsilonNext.get(i));
		}
	}
	@Override
	public void clear() {
		if(output!=null)
		{
			NDArrayCache.store(output);
			output=null;
			NDArrayCache.store(index);
			index=null;
			for(INDArray tmp:epsilonNext)
			NDArrayCache.store(tmp);
			epsilonNext.clear();
			for(Expr e : inputs){
				e.clear();
			}
		}
	}

	@Override
	public int[] shape() {
		return inputs.get(0).shape();
	}
	


}
