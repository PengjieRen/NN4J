package nn4j.utils;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NDArrayCache {

	private static Map<String, Stack<INDArray>> cache = new HashMap<String, Stack<INDArray>>();
	private static Set<INDArray> sets = new HashSet<INDArray>();
	private static Object _lock = new Object();
	private static int limit = 1000000;

	private static String toID(int... shape) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < shape.length; i++)
			sb.append(shape[i] + "-");
		return sb.toString();
	}

	private static String toID(INDArray array) {
		return toID(array.shape());
	}
	
	public static INDArray create(List<INDArray> arrays,int...shape){
		return Nd4j.create(arrays, new int[]{arrays.size(),shape[0],shape[1]},'c');
	}

	public static INDArray get(int... shape) {
		String id = toID(shape);
		synchronized (_lock) {
			if (cache.containsKey(id) && cache.get(id).size() > 0) {
				INDArray a = cache.get(id).pop();
				sets.remove(a);
				return a;
			}
		}
		return Nd4j.zeros(shape,'c');
	}

	public static void store(INDArray array) {
		String id = toID(array);
		synchronized (_lock) {
			if (!cache.containsKey(id)) {
				cache.put(id, new Stack<INDArray>());
			}
			Stack<INDArray> stack = cache.get(id);

			if (stack.size() < limit) {
				if (!sets.contains(array)) {
					stack.add(array);
					sets.add(array);
				}
			}
		}
	}
}
