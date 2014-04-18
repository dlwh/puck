package puck.parser.gen;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import puck.parser.RuleSemiring;
import puck.parser.RuleStructure;

public class GreedySegmentationGenRuleMultiply<C, L>  extends SimpleGenRuleMultiply<C, L> {

	public static final int BINARY_NUM_SEGMENTS = 24;
    private Logger logger = LoggerFactory.getLogger(this.getClass().getName());

	public GreedySegmentationGenRuleMultiply(RuleStructure<C, L> structure, boolean directWrite, RuleSemiring semiring) {
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
		logger.info("min unary segment size: "+min);
		logger.info("max unary segment size: "+max);
		return segmentation;
	}

	public List<IndexedBinaryRule<C, L>>[][] segmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
//		return naiveSegmentBinaries(indexedBinaryRules);
		return smartSegmentBinaries(indexedBinaryRules);
	}
	
	private List<IndexedBinaryRule<C, L>>[][] naiveSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
		List<IndexedBinaryRule<C, L>>[][] segmentation = new List[BINARY_NUM_SEGMENTS][NUM_SM];
		for (int i=0; i<segmentation.length; ++i) {
			for (int j=0; j<segmentation[i].length; ++j) {
				segmentation[i][j] = new ArrayList<IndexedBinaryRule<C, L>>();
			}
		}
		List<IndexedBinaryRule<C, L>> sortedBinaryRules = new ArrayList<IndexedBinaryRule<C, L>>(indexedBinaryRules);
		Collections.sort(sortedBinaryRules, new Comparator<IndexedBinaryRule<C, L>>() {
			public int compare(IndexedBinaryRule<C, L> o1, IndexedBinaryRule<C, L> o2) {
				int parent = Integer.compare(o1.rule().parent().gpu(), o2.rule().parent().gpu());
				if(parent != 0) return parent;
				int lhs = Integer.compare(o1.rule().left().gpu(), o2.rule().left().gpu());
				if(lhs != 0) return lhs;
				int rhs = Integer.compare(o1.rule().right().gpu(), o2.rule().right().gpu());
				return rhs;
			}
		});
		int totalNumSubsegments = BINARY_NUM_SEGMENTS*NUM_SM;
		int rulesPerSubsegment = (int) Math.ceil(((double) sortedBinaryRules.size()) / totalNumSubsegments);
		int index = 0;
		for (int i=0; i<segmentation.length; ++i) {
			for (int j=0; j<segmentation[i].length; ++j) {
				for (int k=0; k<rulesPerSubsegment; ++k) {
					if (index < sortedBinaryRules.size()) {
						segmentation[i][j].add(sortedBinaryRules.get(index));
						index++;
					}
				}
			}
		}
        if(index != sortedBinaryRules.size()) throw new RuntimeException();
		return segmentation;
	}
	
	private List<IndexedBinaryRule<C, L>>[][] smartSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
		List<IndexedBinaryRule<C, L>> sortedBinaryRules = new ArrayList<IndexedBinaryRule<C, L>>(indexedBinaryRules);
		Collections.sort(sortedBinaryRules, new Comparator<IndexedBinaryRule<C, L>>() {
			public int compare(IndexedBinaryRule<C, L> o1, IndexedBinaryRule<C, L> o2) {
				int parent = Integer.compare(o1.rule().parent().gpu(), o2.rule().parent().gpu());
				if(parent != 0) return parent;
				int lhs = Integer.compare(o1.rule().left().gpu(), o2.rule().left().gpu());
				if(lhs != 0) return lhs;
				int rhs = Integer.compare(o1.rule().right().gpu(), o2.rule().right().gpu());
				return rhs;
			}
		});
		int rulesPerSegment = (int) Math.ceil(((double) sortedBinaryRules.size()) / BINARY_NUM_SEGMENTS);
		List<IndexedBinaryRule<C, L>>[][] segmentation = new List[BINARY_NUM_SEGMENTS][];
		int index = 0;
		for (int i=0; i<segmentation.length; ++i) {
			List<IndexedBinaryRule<C, L>> segment = new ArrayList<IndexedBinaryRule<C, L>>();
			for (int k=0; k<rulesPerSegment; ++k) {
				if (index < sortedBinaryRules.size()) {
					segment.add(sortedBinaryRules.get(index));
					index++;
				}
			}
			segmentation[i] = modSubsegmentBinariesByParent(segment, NUM_SM);
//			segmentation[i] = balancedSubsegmentBinariesByParent(segment, NUM_SM);
		}
        logger.trace("Done with binary segment.");
		return segmentation;
	}
	
    private Set<Integer> getParentIndices(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
    	Set<Integer> parents = new HashSet<Integer>();
    	for (IndexedBinaryRule<C, L> rule : indexedBinaryRules) {
    		parents.add(rule.parent().gpu());
    	}
    	return parents;
    }

    private List<IndexedBinaryRule<C, L>>[] modSubsegmentBinariesByParent(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSegments) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[numSegments];
    	for (int i=0; i<numSegments; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
    		result[rule.rule().parent().gpu() % numSegments].add(rule);
    	}
    	return result;
    }

}