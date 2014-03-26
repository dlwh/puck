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
	
	public static int MAX_RULES_PER_BINARY_SEGMENT = 2000;
	public static final int MIN_SINGLE_COARSE_PARENT_GROUP_SIZE = 300;
	
	public VariableSizeCoarseParentSymbolSegmentationGenRuleMultiply(RuleStructure<C, L> structure, boolean directWrite, RuleSemiring semiring) {
		super(structure, directWrite, semiring);
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
	
	
	public List<List<IndexedBinaryRule<C, L>>[]> maxSizeSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSubsegments, int maxSize) {
		List<List<IndexedBinaryRule<C, L>>[]> segmentation = new ArrayList<List<IndexedBinaryRule<C, L>>[]>();
		Deque<IndexedBinaryRule<C, L>> allRules = new ArrayDeque<IndexedBinaryRule<C, L>>();
		List<IndexedBinaryRule<C, L>> sortedRules = new ArrayList<IndexedBinaryRule<C, L>>(indexedBinaryRules);
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
		allRules.addAll(sortedRules);
		
		while (!allRules.isEmpty()) {
			List<IndexedBinaryRule<C, L>> segment = new ArrayList<IndexedBinaryRule<C, L>>();
			while (!allRules.isEmpty() && segment.size() < maxSize) {
				segment.add(allRules.pop());
			}
//			segmentation.add(modSubsegmentBinaries(segment, numSubsegments));
//			segmentation.add(equalSizeSubsegmentBinaries(segment, numSubsegments));
			segmentation.add(equalNumParentsSegmentBinaries(segment, numSubsegments));
		}
		return segmentation;
	}
	
    private List<IndexedBinaryRule<C, L>>[] modSubsegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSegments) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[numSegments];
    	for (int i=0; i<numSegments; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
    		result[rule.rule().parent().gpu() % numSegments].add(rule);
    	}
		for (int i=0; i<numSegments; ++i) {
			System.out.println("subsegment size: "+result[i].size());
		}
    	return result;
    }
    
    private List<IndexedBinaryRule<C, L>>[] equalSizeSubsegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSegments) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[numSegments];
    	for (int i=0; i<numSegments; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	
    	int maxSizePerSubsegment = (int) Math.floor(((double) indexedBinaryRules.size()) / numSegments);
		Deque<IndexedBinaryRule<C, L>> allRules = new ArrayDeque<IndexedBinaryRule<C, L>>();
		allRules.addAll(indexedBinaryRules);
		for (int i=0; i<numSegments; ++i) {
			while (result[i].size() < maxSizePerSubsegment) {
				result[i].add(allRules.pop());
			}
		}
		int index=0;
		while (!allRules.isEmpty()) {
			result[index].add(allRules.pop());
			index++;
		}
		for (int i=0; i<numSegments; ++i) {
			System.out.println("subsegment size: "+result[i].size());
		}
		return result;
    }
    
	private List<IndexedBinaryRule<C, L>>[] equalNumParentsSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSegments) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[numSegments];
    	for (int i=0; i<numSegments; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	
		Map<Integer,List<IndexedBinaryRule<C, L>>> rulesByParent = new HashMap<Integer,List<IndexedBinaryRule<C, L>>>();
		for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
			int parent = rule.parent().gpu();
			List<IndexedBinaryRule<C, L>> rules = rulesByParent.get(parent);
			if (rules == null) {
				rules = new ArrayList<IndexedBinaryRule<C, L>>();
				rulesByParent.put(parent, rules);
			}
			rules.add(rule);
		}
		List<List<IndexedBinaryRule<C, L>>> rulesSegmentedByParent = new ArrayList<List<IndexedBinaryRule<C, L>>>();
		for (Map.Entry<Integer,List<IndexedBinaryRule<C, L>>> entry : rulesByParent.entrySet()) {
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
		while(!rulesSegmentedByParent.isEmpty()) {
			for (int i=0; i<numSegments; ++i) {
				if (!rulesSegmentedByParent.isEmpty()) {
					result[i].addAll(rulesSegmentedByParent.remove(0));
				}
			}
		}
		for (int i=0; i<numSegments; ++i) {
			System.out.println("subsegment size: "+result[i].size());
		}
		return result;
	}
    
}
