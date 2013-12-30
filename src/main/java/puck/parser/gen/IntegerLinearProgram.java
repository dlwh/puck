package puck.parser.gen;

import java.util.Arrays;
import java.util.List;

public abstract class IntegerLinearProgram {

	public abstract int addBoundedIntVar(double lower, double upper);
	
	public abstract int addBoundedVar(double lower, double upper);

	public abstract void addLessThanConstraint(int[] indices, double[] weights, double rhs);

	public abstract void addObjectiveWeight(int pos, double val);

	public abstract void setToMaximize();

	public abstract void lockVariableCount();

	public abstract void optimize();

	public abstract double objectiveValue();
	
	public abstract void clear();

	public abstract double[] solution();
	
	public void addObjectiveWeights(List<Integer> indices, List<Double> weights) {
		for (int i = 0; i < indices.size(); i++) {
			addObjectiveWeight(indices.get(i), weights.get(i));
		}
	}
	
	public void addObjectiveWeights(int[] indices, double[] weights) {
		for (int i = 0; i < indices.length; i++) {
			addObjectiveWeight(indices[i], weights[i]);
		}
	}

	public void addEqualityConstraint(int var, double weight, double rhs) {
		int[] vars = new int[1];
		double[] weights = new double[1];
		vars[0] = var;
		weights[0] = weight;
		addEqualityConstraint(vars, weights, rhs);
	}

	public void addEqualityConstraint(int[] indices, double[] weights, double rhs) {
		addLessThanConstraint(indices, weights, rhs);
		addGreaterThanConstraint(indices, weights, rhs);
	}
	
	public void addGreaterThanConstraint(int[] indices, double[] weights, double rhs) {
		double[] negWeights = new double[weights.length];
		System.arraycopy(weights, 0, negWeights, 0, weights.length);
		for (int i=0; i<negWeights.length; ++i) negWeights[i] *= -1.0;
		addLessThanConstraint(indices, negWeights, -rhs);
	}

	public void addLessThanConstraint(int var, double weight, double rhs) {
		int[] vars = new int[1];
		double[] weights = new double[1];
		vars[0] = var;
		weights[0] = weight;
		addLessThanConstraint(vars, weights, rhs);
	}
	
	public void addGreaterThanConstraint(int var, double weight, double rhs) {
		int[] vars = new int[1];
		double[] weights = new double[1];
		vars[0] = var;
		weights[0] = weight;
		addGreaterThanConstraint(vars, weights, rhs);
	}

	public void addBoundedIntVars(int k, double lower, double upper) {
		for (int i=0; i<k; i++) {
			addBoundedIntVar(lower, upper);
		}
	}

	public void addBoundedVars(int k, double lower, double upper) {
		for (int i=0; i<k; i++) {
			addBoundedVar(lower, upper);
		}
	}

	public void addOrConstraint(int orIndex, int[] indices) {
		addOrConstraintLeft(orIndex, indices);
		addOrConstraintRight(orIndex, indices);
	}

	private void addOrConstraintLeft(int orIndex, int[] indices) {
		double[] ifthen = new double[] { 1.0, -1.0 };
		for (int i : indices) {
			int[] pair = new int[] { i, orIndex };
			addLessThanConstraint(pair, ifthen, 0);
		}
	}

	private void addOrConstraintRight(int orIndex, int[] indices) {
		double[] thenif = new double[indices.length + 1];
		thenif[0] = 1.0;
		Arrays.fill(thenif, 1, thenif.length, -1.0);
		int[] all = new int[indices.length + 1];
		all[0] = orIndex;
		System.arraycopy(indices, 0, all, 1, indices.length);
		addLessThanConstraint(all, thenif, 0);
	}

	public void addAndConstraint(int andIndex, int[] indices) {
		addAndConstraintLeft(andIndex, indices);
		addAndConstraintRight(andIndex, indices);
	}

	private void addAndConstraintLeft(int andIndex, int[] indices) {
		double[] ifthen = new double[] { 1.0, -1.0 };
		for (int i : indices) {
			int[] pair = new int[] { andIndex, i };
			addLessThanConstraint(pair, ifthen, 0);
		}
	}

	private void addAndConstraintRight(int andIndex, int[] indices) {
		double[] thenif = new double[indices.length + 1];
		thenif[0] = -1.0;
		Arrays.fill(thenif, 1, thenif.length, 1.0);
		int[] all = new int[indices.length + 1];
		all[0] = andIndex;
		System.arraycopy(indices, 0, all, 1, indices.length);
		addLessThanConstraint(all, thenif, indices.length - 1);
	}

}