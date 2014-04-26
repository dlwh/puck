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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import puck.parser.RuleSemiring;
import puck.parser.RuleStructure;

public class CoarseParentSymbolGrammarClusterer<C, L> implements GrammarClusterer<C, L>{

    private Logger logger = LoggerFactory.getLogger(this.getClass().getName());

    public static final int MIN_SINGLE_COARSE_PARENT_GROUP_SIZE = 300;

	public List<IndexedUnaryRule<C, L>>[] segmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules) {
		List<IndexedUnaryRule<C, L>>[] segmentation = new List[] {indexedUnaryRules};
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for (List segment : segmentation) {
			min = Math.min(segment.size(), min);
			max = Math.max(segment.size(), max);
		}
		logger.info("min unary segment size: "+min);
		logger.info("max unary segment size: "+max);
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

        if(logger.isTraceEnabled())
            for (List<IndexedBinaryRule<C, L>> list : rulesSegmentedByParent) logger.trace("Segment size: " + list.size());

		List<List<IndexedBinaryRule<C, L>>[]> segmentation = new ArrayList<List<IndexedBinaryRule<C, L>>[]>();
		while(!rulesSegmentedByParent.isEmpty()) {
			List<IndexedBinaryRule<C, L>> segment = rulesSegmentedByParent.remove(0);
			if (segment.size() >= MIN_SINGLE_COARSE_PARENT_GROUP_SIZE) {
//				segmentation.add(modSubsegmentBinaries(segment, NUM_SM));
//				segmentation.add(equalSizeSubsegmentBinaries(segment, NUM_SM));
//                List<IndexedBinaryRule<C, L>>[] subSegmentation = equalNumParentsSegmentBinaries(segment, NUM_SM);
                List<IndexedBinaryRule<C, L>>[] subSegmentation = equalSizeSubsegmentBinaries(segment, NUM_SM);
                List<List<IndexedBinaryRule<C, L>>> outSegmentation = new ArrayList<>();
                for(List<IndexedBinaryRule<C, L>> s: subSegmentation) {
                    if(!s.isEmpty()) {
                        outSegmentation.add(s);
                    }
                }
                if(!outSegmentation.isEmpty())
                    segmentation.add(outSegmentation.toArray(new List[0]));
			} else {
				for (List<IndexedBinaryRule<C, L>> nextSegment : rulesSegmentedByParent) {
					segment.addAll(nextSegment);
				}
//				segmentation.add(modSubsegmentBinaries(segment, NUM_SM));
//				segmentation.add(equalSizeSubsegmentBinaries(segment, NUM_SM));

//                List<IndexedBinaryRule<C, L>>[] subSegmentation = equalNumParentsSegmentBinaries(segment, NUM_SM);
                List<IndexedBinaryRule<C, L>>[] subSegmentation = equalSizeSubsegmentBinaries(segment, NUM_SM);
                List<List<IndexedBinaryRule<C, L>>> outSegmentation = new ArrayList<>();
                for(List<IndexedBinaryRule<C, L>> s: subSegmentation) {
                    if(!s.isEmpty()) {
                        outSegmentation.add(s);
                    }
                }
                if(!outSegmentation.isEmpty())
                    segmentation.add(outSegmentation.toArray(new List[0]));
				break;
			}
		}
		
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (List[] segment : segmentation) {
            double localMin = Double.POSITIVE_INFINITY;
            double localMax = Double.NEGATIVE_INFINITY;
            for (List sub : segment) {
                min = Math.min(sub.size(), min);
                localMin = Math.min(sub.size(), localMin);
                max = Math.max(sub.size(), max);
                localMax = Math.max(sub.size(), localMax);
            }
            logger.info("local min binary sub segment size: " + localMin + " " + localMax);
        }
        logger.info("min binary sub segment size: " + min);
        logger.info("max binary sub segment size: " + max);
		return segmentation.toArray(new List[0][0]);
	}
	
    private List<IndexedBinaryRule<C, L>>[] modSubsegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSegments) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[numSegments];
    	for (int i=0; i<numSegments; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
    		result[rule.rule().parent().gpu() % numSegments].add(rule);
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
		return result;
	}
	
}
