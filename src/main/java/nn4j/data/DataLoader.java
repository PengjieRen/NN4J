package nn4j.data;

import java.util.List;

public abstract class DataLoader {

	public abstract Batch next();
	
	public abstract boolean hasNext();
	
	public abstract void reset();
	
	public abstract List<Data> data();
	
}
