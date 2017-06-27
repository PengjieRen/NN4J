package nn4j.cg;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.factory.Nd4j;

import nn4j.expr.DefaultParamInitializer;
import nn4j.expr.Expr;
import nn4j.expr.Merge;
import nn4j.expr.Parameter;
import nn4j.expr.Parameter.RegType;
import nn4j.expr.Parameter.Updater;
import nn4j.expr.ParameterManager;
import nn4j.expr.WeightInit;
import nn4j.utils.VocabHolder;

public class LookupTable {

	private INDArray mem;
	private Map<String, Parameter> name_param;
	private int dimension;
	private RegType type;
	private float lambdaReg;
	private boolean updatable;
	private ParameterManager pm;
	private Updater updater;

	public LookupTable(ParameterManager pm, int count, int dimension, RegType type, float lambdaReg,
			boolean updatable) {
		this.pm = pm;
		this.dimension = dimension;
		this.type = type;
		this.lambdaReg = lambdaReg;
		this.updatable = updatable;
		mem = Nd4j.zeros(count, dimension);
		name_param = new HashMap<String, Parameter>();
		this.updater = pm.getUpdater();
	}

	public LookupTable(ParameterManager pm, VocabHolder vocab, RegType type, float lambdaReg, boolean updatable) {
		this.pm = pm;
		this.type = type;
		this.lambdaReg = lambdaReg;
		this.updatable = updatable;
		name_param = new HashMap<String, Parameter>();
		this.updater = pm.getUpdater();

		Set<String> words = vocab.words();
		this.dimension = vocab.dimension();
		mem = Nd4j.zeros(words.size(), dimension);

		for (String w : words) {
			add(w, vocab.toEmbed(w));
		}
	}

	public void add(String name, INDArray value) {
		if (!name_param.containsKey(name)) {
			int row = name_param.size();
			if (value != null) {
				mem.putRow(row, value);
			} else {
				mem.putRow(row, new DefaultParamInitializer(WeightInit.DISTRIBUTION, new NormalDistribution(0, 0.01f))
						.init(new int[] { 1, dimension }));
			}

			if (name.equals("<p>"))
				name_param.put(name, new Parameter(mem.getRow(row), type, lambdaReg, false, Updater.NONE));
			else
				name_param.put(name, new Parameter(mem.getRow(row), type, lambdaReg, updatable, updater));

		}
	}

	public void init(List<String> names) {
		for (String name : names) {
			add(name, null);
		}
	}

	public Expr get(String... names) {
		Parameter[] params = new Parameter[names.length];
		for (int i = 0; i < names.length; i++) {

			if (names[i].equals("<p>")) {
				params[i] = name_param.get(names[i]);
				continue;
			}

			if (name_param.containsKey(names[i]))
				params[i] = pm.createParameter(name_param.get(names[i]), true);
			else
				params[i] = pm.createParameter(name_param.get("<u>"), true);
		}

		return new Merge(params);
	}

	public void save2File(File file) {
		try {
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8"));
			for (String key : name_param.keySet()) {
				bw.write(key + " " + name_param.get(key));
				bw.newLine();
			}
			bw.write(mem.toString());
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
