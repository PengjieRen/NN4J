package nn4j.expr;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;

import nn4j.utils.NDArrayCache;

public class DefaultParamInitializer implements ParamInitializer{

	private Distribution dist;
	private WeightInit init;
	
	public DefaultParamInitializer(WeightInit init,float low,float high){
		this(init,new UniformDistribution(low,high));
	}
	
	
	public DefaultParamInitializer(WeightInit init,Distribution dist){
		this.init=init;
		this.dist=dist;
	}
	
	
	public INDArray init(INDArray view){
		return WeightInitUtil.initWeights(1, 1, view.shape(), init, dist, view);
	}
	public INDArray init(int... shape){
		INDArray view=NDArrayCache.get(shape);
		return WeightInitUtil.initWeights(1, 1, shape, init, dist, view);
	}
}
