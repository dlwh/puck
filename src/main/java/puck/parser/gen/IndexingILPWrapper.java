package puck.parser.gen;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class IndexingILPWrapper {
	
	Map<Object,Integer> objectToIndex;
	IntegerLinearProgram ilp;
	boolean locked;
	
	public IndexingILPWrapper(IntegerLinearProgram ilp) {
		this.ilp = ilp;
		this.objectToIndex = new HashMap<Object,Integer>();
	}
	
	public boolean containsObject(Object obj) {
		return objectToIndex.containsKey(obj);
	}
	
	public int numVars() {
		return objectToIndex.keySet().size();
	}
	
	private int getIndex(Object obj) {
		if (!containsObject(obj)) {
			throw new RuntimeException("Object not in indexer.");
		} else {
			return objectToIndex.get(obj);
		}
	}
	
	private int[] getIndices(Object[] objs) {
		int[] indices = new int[objs.length];
		for (int i=0; i<indices.length; ++i) {
			indices[i] = getIndex(objs[i]);
		}
		return indices;
	}
	
	private List<Integer> getIndices(List<Object> objs) {
		List<Integer> indices = new ArrayList<Integer>();
		for (Object obj : objs) {
			indices.add(getIndex(obj));
		}
		return indices;
	}
	
	public int addBoundedIntVar(Object obj, double lower, double upper) {
		if (containsObject(obj)) {
			throw new RuntimeException("Object already added to indexer.");
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
	
	public int addBoundedVar(Object obj, double lower, double upper) {
		if (containsObject(obj)) {
			throw new RuntimeException("Object already added to indexer.");
		} else {
			int index = ilp.addBoundedVar(lower, upper);
			objectToIndex.put(obj, index);
			return index;
		}
	}

	public void addLessThanConstraint(Object[] objs, double[] weights, double rhs) {
		ilp.addLessThanConstraint(getIndices(objs), weights, rhs);
	}

	public void addObjectiveWeight(Object obj, double val) {
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

	public Map<Object,Double> solution() {
		Map<Object,Double> solution = new HashMap<Object, Double>();
		double[] solutionByIndex = ilp.solution();
		for (Object obj : objectToIndex.keySet()) {
			solution.put(obj, solutionByIndex[getIndex(obj)]);
		}
		return solution;
	}
	
	public void addObjectiveWeights(List<Object> objs, List<Double> weights) {
		ilp.addObjectiveWeights(getIndices(objs), weights);
	}
	
	public void addObjectiveWeights(Object[] objs, double[] weights) {
		ilp.addObjectiveWeights(getIndices(objs), weights);
	}

	public void addEqualityConstraint(Object obj, double weight, double rhs) {
		ilp.addEqualityConstraint(getIndex(obj), weight, rhs);
	}

	public void addEqualityConstraint(Object[] objs, double[] weights, double rhs) {
		ilp.addEqualityConstraint(getIndices(objs), weights, rhs);
	}
	
	public void addGreaterThanConstraint(Object[] objs, double[] weights, double rhs) {
		ilp.addGreaterThanConstraint(getIndices(objs), weights, rhs);
	}

	public void addLessThanConstraint(Object obj, double weight, double rhs) {
		ilp.addLessThanConstraint(getIndex(obj), weight, rhs);
	}
	
	public void addGreaterThanConstraint(Object obj, double weight, double rhs) {
		ilp.addGreaterThanConstraint(getIndex(obj), weight, rhs);
	}

	public void addOrConstraint(Object orObj, Object[] objs) {
		ilp.addOrConstraint(getIndex(orObj), getIndices(objs));
	}

	public void addAndConstraint(Object andObj, Object[] objs) {
		ilp.addAndConstraint(getIndex(andObj), getIndices(objs));
	}
	
}
