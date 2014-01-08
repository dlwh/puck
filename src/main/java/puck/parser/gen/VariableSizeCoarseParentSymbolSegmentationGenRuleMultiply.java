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

import puck.parser.RuleSemiring;
import puck.parser.RuleStructure;

public class VariableSizeCoarseParentSymbolSegmentationGenRuleMultiply<C, L> extends SimpleGenRuleMultiply<C, L> {
	
	public static int MAX_RULES_PER_UNARY_SEGMENT = 1000;
	public static int MAX_RULES_PER_BINARY_SEGMENT = 1000;
	public static final int MIN_SINGLE_COARSE_PARENT_GROUP_SIZE = 300;
	
//	public static final int MAX_BADNESS = 60;

	public VariableSizeCoarseParentSymbolSegmentationGenRuleMultiply(RuleStructure<C, L> structure, boolean directWrite, RuleSemiring semiring) {
		super(structure, directWrite, semiring);
	}

	public List<IndexedUnaryRule<C, L>>[] segmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules) {
		List<IndexedUnaryRule<C, L>>[] segmentation = maxSizeSegmentUnaries(indexedUnaryRules, MAX_RULES_PER_UNARY_SEGMENT);
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
	
	public List<IndexedUnaryRule<C, L>>[] maxSizeSegmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules, int maxSize) {
		List<List<IndexedUnaryRule<C, L>>> segmentation = new ArrayList<List<IndexedUnaryRule<C, L>>>();
		List<IndexedUnaryRule<C, L>> allRules = new ArrayList<IndexedUnaryRule<C, L>>(indexedUnaryRules);
		while (!allRules.isEmpty()) {
			List<IndexedUnaryRule<C, L>> segment = new ArrayList<IndexedUnaryRule<C, L>>();
			while (!allRules.isEmpty() && segment.size() < maxSize) {
				segment.add(allRules.remove(0));
			}
			segmentation.add(segment);
		}
		return segmentation.toArray(new List[0]);
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
				segmentation.addAll(maxSizeSegmentBinaries(segment, NUM_SM, MAX_RULES_PER_BINARY_SEGMENT));
			} else {
				for (List<IndexedBinaryRule<C, L>> nextSegment : rulesSegmentedByParent) {
					segment.addAll(nextSegment);
				}
				segmentation.addAll(maxSizeSegmentBinaries(segment, NUM_SM, MAX_RULES_PER_BINARY_SEGMENT));
				break;
			}
		}
		
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (List[] segment : segmentation) {
            for (List sub : segment) {
                min = Math.min(sub.size(), min);
                max = Math.max(sub.size(), max);
            }
        }
        System.out.println("min binary sub segment size: "+min);
        System.out.println("max binary sub segment size: "+max);
		return segmentation.toArray(new List[0][0]);
	}
	
//	public List<List<IndexedBinaryRule<C, L>>[]> maxSizeSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSubsegments, int maxSize) {
//		List<List<IndexedBinaryRule<C, L>>[]> segmentation = new ArrayList<List<IndexedBinaryRule<C, L>>[]>();
//		Deque<IndexedBinaryRule<C, L>> allRules = new ArrayDeque<IndexedBinaryRule<C, L>>();
//		List<IndexedBinaryRule<C, L>> sortedRules = new ArrayList<IndexedBinaryRule<C, L>>(indexedBinaryRules);
//		Collections.sort(sortedRules, new Comparator<IndexedBinaryRule<C, L>>() {
//			public int compare(IndexedBinaryRule<C, L> o1, IndexedBinaryRule<C, L> o2) {
//				int parent = Integer.compare(o1.rule().parent().gpu(), o2.rule().parent().gpu());
//				if(parent != 0) return parent;
//				int lhs = Integer.compare(o1.rule().left().gpu(), o2.rule().left().gpu());
//				if(lhs != 0) return lhs;
//				int rhs = Integer.compare(o1.rule().right().gpu(), o2.rule().right().gpu());
//				return rhs;
//			}
//		});
//		allRules.addAll(sortedRules);
//		
//		while (!allRules.isEmpty()) {
//	          List<IndexedBinaryRule<C, L>>[] segment = new List[numSubsegments];
//	          for(int sub = 0; sub < numSubsegments; sub++) {
//	        	  segment[sub] = new ArrayList<IndexedBinaryRule<C, L>>();
//	        	  while (!allRules.isEmpty() && segment[sub].size() < MAX_RULES_PER_BINARY_SEGMENT / numSubsegments) {
//	        		  segment[sub].add(allRules.pop());
//	        	  }
//	          }
//	          segmentation.add(segment);
//		}
//		return segmentation;
//	}
	
	public List<List<IndexedBinaryRule<C, L>>[]> maxSizeSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSubsegments, int maxSize) {
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
		
		while (!allRules.isEmpty()) {
			List<IndexedBinaryRule<C, L>> segment = new ArrayList<IndexedBinaryRule<C, L>>();
			while (!allRules.isEmpty() && segment.size() < MAX_RULES_PER_BINARY_SEGMENT) {
				segment.add(allRules.pop());
			}
			segmentation.add(modSubsegmentBinaries(segment, NUM_SM));
		}
		return segmentation;
	}
	
    private List<IndexedBinaryRule<C, L>>[] modSubsegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSegments) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[numSegments];
    	for (int i=0; i<numSegments; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
    		result[rule.rule().parent().gpu() % numSegments].add(rule);
    	}
    	return result;
    }
	
//	private List<List<IndexedBinaryRule<C, L>>[]> variableSizeSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSubsegments) {
//		List<List<IndexedBinaryRule<C, L>>[]> segmentation = new ArrayList<List<IndexedBinaryRule<C, L>>[]>();
//        Deque<IndexedBinaryRule<C, L>> allRules = new ArrayDeque<IndexedBinaryRule<C, L>>();
//        List<IndexedBinaryRule<C, L>> sortedRules = new ArrayList<IndexedBinaryRule<C, L>>(indexedBinaryRules);
//        Collections.sort(sortedRules, new Comparator<IndexedBinaryRule<C, L>>() {
//            public int compare(IndexedBinaryRule<C, L> o1, IndexedBinaryRule<C, L> o2) {
//                int parent = Integer.compare(o1.rule().parent().gpu(), o2.rule().parent().gpu());
//                if(parent != 0) return parent;
//                int lhs = Integer.compare(o1.rule().left().gpu(), o2.rule().left().gpu());
//                if(lhs != 0) return lhs;
//                int rhs = Integer.compare(o1.rule().right().gpu(), o2.rule().right().gpu());
//                return rhs;
//            }
//        });
//
//        allRules.addAll(sortedRules);
//
//        while(!allRules.isEmpty()) {
//            List<IndexedBinaryRule<C, L>>[] segment = new List[numSubsegments];
//            for(int sub = 0; sub < numSubsegments; sub++) {
//                segment[sub] = new ArrayList<IndexedBinaryRule<C, L>>();
//            }
//            segmentation.add(segment);
//
//            // use parent only once per segment
////            Set<Integer> usedParents = new HashSet<Integer>();
////            List<IndexedBinaryRule<C, L>> skippedRules = new ArrayList<IndexedBinaryRule<C, L>>();
//
//            for(int sub = 0; sub < numSubsegments; sub++) {
//                List<IndexedBinaryRule<C, L>> subseg = segment[sub];
//                Set<Integer> parents = new HashSet<Integer>();
//                Set<Integer> lefts = new HashSet<Integer>();
//                Set<Integer> rights = new HashSet<Integer>();
//
//                while(!allRules.isEmpty() && badness(subseg, parents, lefts, rights) < MAX_BADNESS) {
//                    IndexedBinaryRule<C, L> rule = allRules.pop();
////                    if(!usedParents.contains(rule.parent().gpu())) {
//                        parents.add(rule.parent().gpu());
//                        lefts.add(rule.rule().left().gpu());
//                        rights.add(rule.rule().right().gpu());
//                        subseg.add(rule);
////                    } else {
////                        skippedRules.add(rule);
////                    }
//                }
//
////                usedParents.addAll(parents);
//            }
//
////            Collections.reverse(skippedRules);
////            for(IndexedBinaryRule<C, L> r: skippedRules)
////                allRules.push(r);
//        }
//
//        return segmentation;
//    }
//	
//    private int badness(List<IndexedBinaryRule<C, L>> rules, Set<Integer> parents, Set<Integer> left, Set<Integer> right) {
//        return left.size() + right.size();
//    }
    
}
