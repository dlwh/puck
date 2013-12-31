package puck.parser.gen;

import java.util.ArrayList;
import java.util.List;

import puck.parser.RuleStructure;

public class CoarseParentSymbolSegmentationGenRuleMultiply<C, L> extends SimpleGenRuleMultiply<C, L> {

	public CoarseParentSymbolSegmentationGenRuleMultiply(RuleStructure<C, L> structure) {
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
		return null;
	}
	
    private List<IndexedBinaryRule<C, L>>[] modSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int m) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[m];
    	for (int i=0; i<m; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
    		result[rule.rule().parent().gpu() % m].add(rule);
    	}
    	return result;
    }

}
