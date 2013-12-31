package puck.parser.gen;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class IndexingILPWrapper<T> {
	
	Map<T,Integer> objectToIndex;
	IntegerLinearProgram ilp;
	boolean locked;
	
	public IndexingILPWrapper(IntegerLinearProgram ilp) {
		this.ilp = ilp;
		this.objectToIndex = new HashMap<T,Integer>();
	}
	
	public boolean containsObject(T obj) {
		return objectToIndex.containsKey(obj);
	}
	
	public int numVars() {
		return objectToIndex.keySet().size();
	}
	
	private int getIndex(T obj) {
		if (!containsObject(obj)) {
			throw new RuntimeException("T not in indexer.");
		} else {
			return objectToIndex.get(obj);
		}
	}
	
	private int[] getIndices(T[] objs) {
		int[] indices = new int[objs.length];
		for (int i=0; i<indices.length; ++i) {
			indices[i] = getIndex(objs[i]);
		}
		return indices;
	}
	
	private List<Integer> getIndices(List<T> objs) {
		List<Integer> indices = new ArrayList<Integer>();
		for (T obj : objs) {
			indices.add(getIndex(obj));
		}
		return indices;
	}
	
	public int addBoundedIntVar(T obj, double lower, double upper) {
		if (containsObject(obj)) {
			throw new RuntimeException("T already added to indexer.");
		} else {
			int index = ilp.addBoundedIntVar(lower, upper);
			objectToIndex.put(obj, index);
			return index;
		}
	}
	
	public void lockVariableCount() {
		ilp.lockVariableCount();
	}

	public void optimize() {
		ilp.optimize();
	}
	
	public int addBoundedVar(T obj, double lower, double upper) {
		if (containsObject(obj)) {
			throw new RuntimeException("T already added to indexer.");
		} else {
			int index = ilp.addBoundedVar(lower, upper);
			objectToIndex.put(obj, index);
			return index;
		}
	}

	public void addLessThanConstraint(T[] objs, double[] weights, double rhs) {
		ilp.addLessThanConstraint(getIndices(objs), weights, rhs);
	}

	public void addObjectiveWeight(T obj, double val) {
		ilp.addObjectiveWeight(getIndex(obj), val);
	}

	public void setToMaximize() {
		ilp.setToMaximize();
	}

	public double objectiveValue() {
		return ilp.objectiveValue();
	}
	
	public void clear() {
		ilp.clear();
	}

	public Map<T,Double> solution() {
		Map<T,Double> solution = new HashMap<T, Double>();
		double[] solutionByIndex = ilp.solution();
		for (T obj : objectToIndex.keySet()) {
			solution.put(obj, solutionByIndex[getIndex(obj)]);
		}
		return solution;
	}
	
	public void addObjectiveWeights(List<T> objs, List<Double> weights) {
		ilp.addObjectiveWeights(getIndices(objs), weights);
	}
	
	public void addObjectiveWeights(T[] objs, double[] weights) {
		ilp.addObjectiveWeights(getIndices(objs), weights);
	}

	public void addEqualityConstraint(T obj, double weight, double rhs) {
		ilp.addEqualityConstraint(getIndex(obj), weight, rhs);
	}

	public void addEqualityConstraint(T[] objs, double[] weights, double rhs) {
		ilp.addEqualityConstraint(getIndices(objs), weights, rhs);
	}
	
	public void addGreaterThanConstraint(T[] objs, double[] weights, double rhs) {
		ilp.addGreaterThanConstraint(getIndices(objs), weights, rhs);
	}

	public void addLessThanConstraint(T obj, double weight, double rhs) {
		ilp.addLessThanConstraint(getIndex(obj), weight, rhs);
	}
	
	public void addGreaterThanConstraint(T obj, double weight, double rhs) {
		ilp.addGreaterThanConstraint(getIndex(obj), weight, rhs);
	}

	public void addOrConstraint(T orObj, T[] objs) {
		ilp.addOrConstraint(getIndex(orObj), getIndices(objs));
	}

	public void addAndConstraint(T andObj, T[] objs) {
		ilp.addAndConstraint(getIndex(andObj), getIndices(objs));
	}
	
}
