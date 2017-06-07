package nn4j.data;

import java.util.List;

import nn4j.expr.ParameterManager;

public abstract class DataLoader {

	protected ParameterManager pm;

	protected DataLoader(ParameterManager pm) {
		this.pm = pm;
	}

	public abstract Batch next();

	public abstract boolean hasNext();

	public abstract void reset();

	public abstract List<Data> data();

}
