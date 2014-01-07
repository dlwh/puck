package puck.parser.gen;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import puck.parser.RuleStructure;

public class VariableSizeCoarseParentSymbolSegmentationGenRuleMultiply<C, L> extends SimpleGenRuleMultiply<C, L> {
	
	public static final int MIN_SINGLE_COARSE_PARENT_GROUP_SIZE = 300;
	public static final int MAX_BADNESS = 120;

	public VariableSizeCoarseParentSymbolSegmentationGenRuleMultiply(RuleStructure<C, L> structure, boolean directWrite) {
		super(structure, directWrite);
	}

	public List<IndexedUnaryRule<C, L>>[] segmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules) {
		List<IndexedUnaryRule<C, L>>[] segmentation = new List[] {indexedUnaryRules};
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for (List segment : segmentation) {
			min = Math.min(segment.size(), min);
			max = Math.max(segment.size(), max);
		}
		System.out.println("min unary segment size: "+min);
		System.out.println("max unary segment size: "+max);
		return segmentation;
	}

	public List<IndexedBinaryRule<C, L>>[][] segmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
		Map<Integer,List<IndexedBinaryRule<C, L>>> rulesByCoarseParent = new HashMap<Integer,List<IndexedBinaryRule<C, L>>>();
		for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
			int coarseParent = rule.parent().coarse();
			List<IndexedBinaryRule<C, L>> rules = rulesByCoarseParent.get(coarseParent);
			if (rules == null) {
				rules = new ArrayList<IndexedBinaryRule<C, L>>();
				rulesByCoarseParent.put(coarseParent, rules);
			}
			rules.add(rule);
		}
		List<List<IndexedBinaryRule<C, L>>> rulesSegmentedByParent = new ArrayList<List<IndexedBinaryRule<C, L>>>();
		for (Map.Entry<Integer,List<IndexedBinaryRule<C, L>>> entry : rulesByCoarseParent.entrySet()) {
			rulesSegmentedByParent.add(entry.getValue());
		}
		Collections.sort(rulesSegmentedByParent, new Comparator<List<IndexedBinaryRule<C, L>>>() {
			public int compare(List<IndexedBinaryRule<C, L>> o1, List<IndexedBinaryRule<C, L>> o2) {
				if (o1.size() > o2.size()) {
					return -1;
				} else if (o1.size() < o2.size()) {
					return 1;
				} else {
					return 0;
				}
			}
		});
		for (List<IndexedBinaryRule<C, L>> list : rulesSegmentedByParent) System.out.println(list.size());
		List<List<IndexedBinaryRule<C, L>>[]> segmentation = new ArrayList<List<IndexedBinaryRule<C, L>>[]>();
		while(!rulesSegmentedByParent.isEmpty()) {
			List<IndexedBinaryRule<C, L>> segment = rulesSegmentedByParent.remove(0);
			if (segment.size() >= MIN_SINGLE_COARSE_PARENT_GROUP_SIZE) {
				segmentation.addAll(variableSizeSubsegmentBinaries(segment, NUM_SM));
			} else {
				for (List<IndexedBinaryRule<C, L>> nextSegment : rulesSegmentedByParent) {
					segment.addAll(nextSegment);
				}
				segmentation.addAll(variableSizeSubsegmentBinaries(segment, NUM_SM));
				break;
			}
		}
		System.out.println("Done with binary segment.");
		return segmentation.toArray(new List[0][0]);
	}
	
	private List<List<IndexedBinaryRule<C, L>>[]> variableSizeSubsegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSubsegments) {
		List<List<IndexedBinaryRule<C, L>>[]> segmentation = new ArrayList<List<IndexedBinaryRule<C, L>>[]>();
        Deque<IndexedBinaryRule<C, L>> allRules = new ArrayDeque<IndexedBinaryRule<C, L>>();
        List<IndexedBinaryRule<C, L>> sortedRules = new ArrayList<IndexedBinaryRule<C, L>>(indexedBinaryRules);
        Collections.sort(sortedRules, new Comparator<IndexedBinaryRule<C, L>>() {
            public int compare(IndexedBinaryRule<C, L> o1, IndexedBinaryRule<C, L> o2) {
                int parent = Integer.compare(o1.rule().parent().gpu(), o2.rule().parent().gpu());
                if(parent != 0) return parent;
                int lhs = Integer.compare(o1.rule().left().gpu(), o2.rule().left().gpu());
                if(lhs != 0) return lhs;
                int rhs = Integer.compare(o1.rule().right().gpu(), o2.rule().right().gpu());
                return rhs;
            }
        });

        allRules.addAll(sortedRules);

        while(!allRules.isEmpty()) {
            List<IndexedBinaryRule<C, L>>[] segment = new List[numSubsegments];
            for(int sub = 0; sub < numSubsegments; sub++) {
                segment[sub] = new ArrayList<IndexedBinaryRule<C, L>>();
            }
            segmentation.add(segment);

            // use parent only once per segment
//            Set<Integer> usedParents = new HashSet<Integer>();
//            List<IndexedBinaryRule<C, L>> skippedRules = new ArrayList<IndexedBinaryRule<C, L>>();

            for(int sub = 0; sub < numSubsegments; sub++) {
                List<IndexedBinaryRule<C, L>> subseg = segment[sub];
                Set<Integer> parents = new HashSet<Integer>();
                Set<Integer> lefts = new HashSet<Integer>();
                Set<Integer> rights = new HashSet<Integer>();

                while(!allRules.isEmpty() && badness(subseg, parents, lefts, rights) < MAX_BADNESS) {
                    IndexedBinaryRule<C, L> rule = allRules.pop();
//                    if(!usedParents.contains(rule.parent().gpu())) {
                        parents.add(rule.parent().gpu());
                        lefts.add(rule.rule().left().gpu());
                        rights.add(rule.rule().right().gpu());
                        subseg.add(rule);
//                    } else {
//                        skippedRules.add(rule);
//                    }
                }

//                usedParents.addAll(parents);
            }

//            Collections.reverse(skippedRules);
//            for(IndexedBinaryRule<C, L> r: skippedRules)
//                allRules.push(r);
        }

        return segmentation;
    }
	
    private int badness(List<IndexedBinaryRule<C, L>> rules, Set<Integer> parents, Set<Integer> left, Set<Integer> right) {
        return left.size() + right.size();
    }
	
//    private List<IndexedBinaryRule<C, L>>[][] balancedSubsegmentBinariesByParent(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSubsegments, int maxBadness, int maxSegmentSize, int maxNumSegmnets) {
//    	Set<Integer> allParents = getParentIndices(indexedBinaryRules); 
//    	Set<Integer> allLefts = getLeftChildIndices(indexedBinaryRules); 
//    	IndexingILPWrapper<String> ilp = new IndexingILPWrapper<String>(new CPLEXIntegerLinearProgram(20, 8, false));
//    	ilp.addBoundedVar("maxSegmentSize", 0, Integer.MAX_VALUE);
//    	ilp.addBoundedVar("maxActiveParents", 0, Integer.MAX_VALUE);
//    	for (int segmentId=0; segmentId<numSegments; ++segmentId) {
//    		for (int ruleId=0; ruleId<indexedBinaryRules.size(); ++ruleId) {
//    			ilp.addBoundedIntVar("ruleAssignment"+segmentId+"_"+ruleId, 0, 1);
//    		}
//    		for (Integer parentIndex : allParents) {
//    			ilp.addBoundedIntVar("parentAssignment"+segmentId+"_"+parentIndex, 0, 1);
//    		}
//    	}
//    	ilp.lockVariableCount();
//
//    	// objective
//    	ilp.addObjectiveWeights(new String[] {"maxSegmentSize", "maxActiveParents"}, new double[] {1.0, 1.0});
//
//    	// maxSegmentSize is greater than all segment sizes
//    	for (int segmentId=0; segmentId<numSegments; ++segmentId) {
//    		String[] objs = new String[indexedBinaryRules.size()+1];
//    		double[] weights = new double[indexedBinaryRules.size()+1];
//    		objs[0] = "maxSegmentSize";
//    		weights[0] = 1.0;
//    		for (int ruleId=0; ruleId<indexedBinaryRules.size(); ++ruleId) {
//    			objs[ruleId+1] = "ruleAssignment"+segmentId+"_"+ruleId;
//    			weights[ruleId+1] = -1.0;
//    		}
//    		ilp.addGreaterThanConstraint(objs, weights, 0.0);
//    	}
//    	
//    	// maxActiveParents is greater than num active parents in each segment
//    	for (int segmentId=0; segmentId<numSegments; ++segmentId) {
//    		String[] objs = new String[allParents.size()+1];
//    		double[] weights = new double[allParents.size()+1];
//    		objs[0] = "maxActiveParents";
//    		weights[0] = 1.0;
//    		int index = 1;
//    		for (Integer parentIndex : allParents) {
//    			objs[index] = "parentAssignment"+segmentId+"_"+parentIndex;
//    			weights[index] = -1.0;
//    			index++;
//    		}
//    		ilp.addGreaterThanConstraint(objs, weights, 0.0);
//    	}
//    	
//    	// rule on implies parent on
//    	for (int segmentId=0; segmentId<numSegments; ++segmentId) {
//    		for (int ruleId=0; ruleId<indexedBinaryRules.size(); ++ruleId) {
//    			int parentIndex = indexedBinaryRules.get(ruleId).parent().gpu();
//    			ilp.addGreaterThanConstraint(new String[] {"parentAssignment"+segmentId+"_"+parentIndex, "ruleAssignment"+segmentId+"_"+ruleId}, new double[] {1.0, -1.0}, 0.0);
//    		}
//    	}
//    	
//    	// rule is in exactly one segment
//    	for (int ruleId=0; ruleId<indexedBinaryRules.size(); ++ruleId) {
//    		String[] objs = new String[numSegments];
//    		double[] weights = new double[numSegments];
//    		for (int segmentId=0; segmentId<numSegments; ++segmentId) {
//    			objs[segmentId] = "ruleAssignment"+segmentId+"_"+ruleId;
//    			weights[segmentId] = 1.0;
//    		}
//    		ilp.addEqualityConstraint(objs, weights, 1.0);
//    	}
//    	
//    	// parent is in at most one segment
//    	for (Integer parentIndex : allParents) {
//    		String[] objs = new String[numSegments];
//    		double[] weights = new double[numSegments];
//    		for (int segmentId=0; segmentId<numSegments; ++segmentId) {
//    			objs[segmentId] = "parentAssignment"+segmentId+"_"+parentIndex;
//    			weights[segmentId] = 1.0;
//    		}
//    		ilp.addLessThanConstraint(objs, weights, 1.0);
//    	}
//    	
//    	// solve
//    	ilp.optimize();
//    	System.out.println("ilp objective value: "+ilp.objectiveValue());
//    	Map<String,Double> solution = ilp.solution();
//    	
//    	return null;
//    }
//    
//    private Set<Integer> getParentIndices(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
//    	Set<Integer> parents = new HashSet<Integer>();
//    	for (IndexedBinaryRule<C, L> rule : indexedBinaryRules) {
//    		parents.add(rule.parent().gpu());
//    	}
//    	return parents;
//    }
//    
//    private Set<Integer> getLeftChildIndices(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
//    	Set<Integer> leftChildren = new HashSet<Integer>();
//    	for (IndexedBinaryRule<C, L> rule : indexedBinaryRules) {
//    		leftChildren.add(rule.rule().left().gpu());
//    	}
//    	return leftChildren;
//    }
    
}
