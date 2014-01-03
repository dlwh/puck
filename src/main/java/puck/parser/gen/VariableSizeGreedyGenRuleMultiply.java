package puck.parser.gen;

import puck.parser.RuleStructure;

import java.util.*;

public class VariableSizeGreedyGenRuleMultiply<C, L>  extends SimpleGenRuleMultiply<C, L> {

	public VariableSizeGreedyGenRuleMultiply(RuleStructure<C, L> structure, boolean directWrite) {
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
        List<IndexedBinaryRule<C, L>>[][]  segmentation = naiveSegmentBinaries(indexedBinaryRules);
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (List[] segment : segmentation) {
            for (List sub : segment) {
                min = Math.min(sub.size(), min);
                max = Math.max(sub.size(), max);
            }
        }
        System.out.println("min binary segment size: "+min);
        System.out.println("max binary segment size: "+max);
        return segmentation;
	}

    private static int MAX_BADNESS = 60;

    private int badness(List<IndexedBinaryRule<C, L>> rules, Set<Integer> parents, Set<Integer> left, Set<Integer> right) {
        return left.size() + right.size();
    }
	
	private List<IndexedBinaryRule<C, L>>[][] naiveSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
		List<List<IndexedBinaryRule<C, L>>[]> segmentation = new ArrayList<>();
        Deque<IndexedBinaryRule<C, L>> allRules = new ArrayDeque<>();
        List<IndexedBinaryRule<C, L>> sortedRules = new ArrayList<>(indexedBinaryRules);
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
                segment[sub] = new ArrayList<>();
            }
            segmentation.add(segment);

            // use parent only once per segment
            Set<Integer> usedParents = new HashSet<>();
            List<IndexedBinaryRule<C, L>> skippedRules = new ArrayList<>();

            for(int sub = 0; sub < NUM_SM; sub++) {
                List<IndexedBinaryRule<C, L>> subseg = segment[sub];
                Set<Integer> parents = new HashSet<>();
                Set<Integer> lefts = new HashSet<>();
                Set<Integer> rights = new HashSet<>();

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

	
    private Set<Integer> getParentIndices(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
    	Set<Integer> parents = new HashSet<Integer>();
    	for (IndexedBinaryRule<C, L> rule : indexedBinaryRules) {
    		parents.add(rule.parent().gpu());
    	}
    	return parents;
    }
    


}

