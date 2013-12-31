package puck.parser.gen;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import puck.parser.RuleStructure;

public class GreedySegmentationGenRuleMultiply<C, L>  extends SimpleGenRuleMultiply<C, L> {

	public static final int BINARY_NUM_SEGMENTS = 24;
	
	public GreedySegmentationGenRuleMultiply(RuleStructure<C, L> structure) {
		super(structure);
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
		int totalNumSubclusters = BINARY_NUM_SEGMENTS*NUM_SM;
		int rulesPerSubluster = (int) Math.ceil(((double) sortedBinaryRules.size()) / totalNumSubclusters);
		int index = 0;
		for (int i=0; i<segmentation.length; ++i) {
			for (int j=0; j<segmentation[i].length; ++j) {
				for (int k=0; k<rulesPerSubluster; ++k) {
					if (index < sortedBinaryRules.size()) {
						segmentation[i][j].add(sortedBinaryRules.get(index));
						index++;
					}
				}
			}
		}
		return segmentation;
	}

}