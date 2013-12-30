package puck.parser.gen;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.util.ArrayList;
import java.util.List;

public class CPLEXIntegerLinearProgram extends IntegerLinearProgram {
	
	int numVars = 0;
	int numConstraints = 0;
	int numThreads = 8;
	boolean relaxIntegerConstraint = false;
	List<IloNumVar> cplexVars;
	List<Double> objectiveWeights;
	boolean varsLocked = false;
	boolean maximize = false;
	IloCplex solver;
	double[] solution;
	double obj;
	boolean optimized = false;
	
	public CPLEXIntegerLinearProgram(int numThreads, boolean relaxIntegerConstraint) {
		this(0, numThreads, relaxIntegerConstraint);
	}
	
	public CPLEXIntegerLinearProgram(int maxTime, int numThreads, boolean relaxIntegerConstraint) {
		this.numThreads = numThreads;
		this.relaxIntegerConstraint = relaxIntegerConstraint;
		cplexVars = new ArrayList<IloNumVar>();
		objectiveWeights = new ArrayList<Double>();
		try {
			solver = new IloCplex();

			solver.setOut(null);
			
			if (maxTime > 0) solver.setParam(IloCplex.DoubleParam.TiLim, maxTime);			

			solver.setParam(IloCplex.IntParam.Threads, numThreads);
			
			solver.setParam(IloCplex.DoubleParam.WorkMem, 128);
			solver.setParam(IloCplex.BooleanParam.MemoryEmphasis, false);
			solver.setParam(IloCplex.IntParam.Probe, 3);
			
//			solver.setParam(IloCplex.DoubleParam.EpInt, 0.2);
//			solver.setParam(IloCplex.DoubleParam.EpGap, 0.2);

//			solver.setParam(IloCplex.IntParam.VarSel, 3);
//			solver.setParam(IloCplex.IntParam.VarSel, 4);
			
//			solver.setParam(IloCplex.IntParam.FracCuts, 2);
//			solver.setParam(IloCplex.IntParam.FlowCovers, 2);
//			solver.setParam(IloCplex.IntParam.FlowPaths, 2);
//			solver.setParam(IloCplex.IntParam.Cliques, 3);
//			solver.setParam(IloCplex.IntParam.Covers, 3);
//			solver.setParam(IloCplex.IntParam.GUBCovers, 2);
//			solver.setParam(IloCplex.IntParam.ImplBd, 2);
//			solver.setParam(IloCplex.IntParam.MCFCuts, 2);
//			solver.setParam(IloCplex.IntParam.MIRCuts, 2);
//			solver.setParam(IloCplex.IntParam.ZeroHalfCuts, 2);
//			solver.setParam(IloCplex.IntParam.DisjCuts, 3);
//			solver.setParam(IloCplex.DoubleParam.CutsFactor, 16.0);
//			solver.setParam(IloCplex.IntParam.AggCutLim, 12);
		} catch (IloException e) {
			e.printStackTrace();
		}
	} 

	public void clear() {
		varsLocked = false;
		try {
			solver.clearModel();
		} catch (IloException e) {
			e.printStackTrace();
		}
		cplexVars = new ArrayList<IloNumVar>();
		objectiveWeights = new ArrayList<Double>();
	}
	
	public int addBoundedIntVar(double lower, double upper) {
		if (varsLocked) throw new RuntimeException("Vars locked.");
		numVars++;
		int index = cplexVars.size();
		try {
			if (relaxIntegerConstraint) {
				cplexVars.add(solver.numVar(lower, upper));
			} else {
				if (lower == 0 && upper == 1.0) {
					cplexVars.add(solver.boolVar());
				} else {
					cplexVars.add(solver.intVar((int) lower, (int) upper));
				}
			}
			objectiveWeights.add(0.0);
		} catch (IloException e) {
			e.printStackTrace();
		}
		return index;
	}
	

	public int addBoundedVar(double lower, double upper) {
		if (varsLocked) throw new RuntimeException("Vars locked.");
		numVars++;
		int index = cplexVars.size();
		try {
			cplexVars.add(solver.numVar(lower, upper));
			objectiveWeights.add(0.0);
		} catch (IloException e) {
			e.printStackTrace();
		}
		return index;
	}


	public void addLessThanConstraint(int[] indices, double[] weights, double rhs) {
		numConstraints++;
		IloNumVar[] vars = new IloNumVar[indices.length];
		for (int i=0; i<indices.length; ++i) {
			if (indices[i] >= cplexVars.size()) throw new RuntimeException(String.format("Var %d hasn't been added yet.", indices[i])); 
			vars[i] = cplexVars.get(indices[i]);
		}
		try {
			solver.addLe(solver.scalProd(vars, weights), rhs);
		} catch (IloException e) {
			e.printStackTrace();
		}
	}

	public void addObjectiveWeight(int index, double val) {
		if (index >= cplexVars.size()) throw new RuntimeException(String.format("Var %d hasn't been added yet.", index));
		objectiveWeights.set(index, val);
	}
	
	public void setToMaximize() {
		maximize = true;
	}

	public void lockVariableCount() {
		varsLocked = true;
	}

	public void optimize() {
		lockVariableCount();
		try {
			IloNumVar[] cplexVarsArray = new IloNumVar[cplexVars.size()];
			double[] objectiveWeightsArray = new double[cplexVars.size()];
			for (int i=0; i<cplexVars.size(); ++i) {
				cplexVarsArray[i] = cplexVars.get(i);
				objectiveWeightsArray[i] = objectiveWeights.get(i);
			}
			objectiveWeights = null;
			if (maximize) {
				solver.addMaximize(solver.scalProd(cplexVarsArray, objectiveWeightsArray));
			} else {
				solver.addMinimize(solver.scalProd(cplexVarsArray, objectiveWeightsArray));
			}
			cplexVarsArray = null;
			objectiveWeightsArray = null;
			
			solver.solve();
			obj = solver.getObjValue();
			solution = solver.getValues(cplexVars.toArray(new IloNumVar[0]));
		} catch (IloException e) {
			e.printStackTrace();
		}
		optimized = true;
	}

	public double objectiveValue() {
		return obj;
	}

	public double[] solution() {
		if (!optimized) optimize();
		return solution;
	}
	
	public static void main(String[] args) {
		IntegerLinearProgram solver = new CPLEXIntegerLinearProgram(2400,  8, false);
		solver.addBoundedIntVars(4, 0.0, 1.0);
		int x1 = 0;
		int x2 = 1;
		int x3 = 2;
		int x4 = 3;

		solver.addEqualityConstraint(new int[] {x1, x3}, new double[] {1.0, -1.0}, 0.0);
		solver.addEqualityConstraint(new int[] {x2, x1}, new double[] {1.0, -1.0}, 0.0);
		
		solver.addObjectiveWeight(x2, -0.50);
		solver.addObjectiveWeight(x3, 1.0);
		
		solver.setToMaximize();
		
		solver.optimize();
		
		System.out.println(solver.objectiveValue());
	}
	
}
