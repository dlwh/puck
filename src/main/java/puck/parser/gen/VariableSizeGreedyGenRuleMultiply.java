package puck.parser.gen;

import puck.parser.RuleStructure;

import java.util.*;

public class VariableSizeGreedyGenRuleMultiply<C, L>  extends SimpleGenRuleMultiply<C, L> {
	
	public static int MAX_RULES_PER_UNARY_SEGMENT = 2000;

	public VariableSizeGreedyGenRuleMultiply(RuleStructure<C, L> structure, boolean directWrite, boolean logSpace) {
		super(structure, directWrite, logSpace);
	}

	public List<IndexedUnaryRule<C, L>>[] segmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules) {
		double numUnaries = indexedUnaryRules.size();
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
        List<IndexedBinaryRule<C, L>>[][]  segmentation = variableSizeSegmentBinaries(indexedBinaryRules);
    
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
        return segmentation;
	}

//	public static final int RULES_PER_SUBSEGMENT = 125;
//	
//	private List<IndexedBinaryRule<C, L>>[][] variableSizeSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
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
//            List<IndexedBinaryRule<C, L>>[] segment = new List[NUM_SM];
//            for(int sub = 0; sub < NUM_SM; sub++) {
//                segment[sub] = new ArrayList<IndexedBinaryRule<C, L>>();
//            }
//            segmentation.add(segment);
//
//            for(int sub = 0; sub < NUM_SM; sub++) {
//                List<IndexedBinaryRule<C, L>> subseg = segment[sub];
//                while(!allRules.isEmpty()) {
//                    IndexedBinaryRule<C, L> rule = allRules.pop();
//                        subseg.add(rule);
//                }
//            }
//        }
//        
//        return segmentation.toArray(new List[0][0]);
//    }
	
    private static int MAX_BADNESS = 60;

    private int badness(List<IndexedBinaryRule<C, L>> rules, Set<Integer> parents, Set<Integer> left, Set<Integer> right) {
        return left.size() + right.size();
    }
	
	private List<IndexedBinaryRule<C, L>>[][] variableSizeSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
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
			List<IndexedBinaryRule<C, L>>[] segment = new List[NUM_SM];
			for(int sub = 0; sub < NUM_SM; sub++) {
				segment[sub] = new ArrayList<IndexedBinaryRule<C, L>>();
			}
			segmentation.add(segment);
			
			// use parent only once per segment
            Set<Integer> usedParents = new HashSet<Integer>();
            List<IndexedBinaryRule<C, L>> skippedRules = new ArrayList<IndexedBinaryRule<C, L>>();
			
			for(int sub = 0; sub < NUM_SM; sub++) {
				List<IndexedBinaryRule<C, L>> subseg = segment[sub];
				Set<Integer> parents = new HashSet<Integer>();
				Set<Integer> lefts = new HashSet<Integer>();
				Set<Integer> rights = new HashSet<Integer>();
				
				while(!allRules.isEmpty() && badness(subseg, parents, lefts, rights) < MAX_BADNESS) {
					IndexedBinaryRule<C, L> rule = allRules.pop();
					if(!usedParents.contains(rule.parent().gpu())) {
						parents.add(rule.parent().gpu());
						lefts.add(rule.rule().left().gpu());
						rights.add(rule.rule().right().gpu());
						subseg.add(rule);
					} else {
						skippedRules.add(rule);
					}
				}
				
                usedParents.addAll(parents);
			}
			
            Collections.reverse(skippedRules);
            for(IndexedBinaryRule<C, L> r: skippedRules)
                allRules.push(r);
		}
		
		return segmentation.toArray(new List[0][0]);
	}

}

