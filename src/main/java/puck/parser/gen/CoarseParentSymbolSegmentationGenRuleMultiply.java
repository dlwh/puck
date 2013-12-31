package puck.parser.gen;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
		List<IndexedBinaryRule<C, L>>[][] segmentation = new List[rulesByCoarseParent.size()][];
		int index=0;
		for (Map.Entry<Integer,List<IndexedBinaryRule<C, L>>> entry : rulesByCoarseParent.entrySet()) {
			segmentation[index] = modSegmentBinaries(entry.getValue(), NUM_SM);
			index++;
		}
		return segmentation;
	}
	
    private List<IndexedBinaryRule<C, L>>[] modSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSegments) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[numSegments];
    	for (int i=0; i<numSegments; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
    		result[rule.rule().parent().gpu() % numSegments].add(rule);
    	}
    	return result;
    }
    
    private Set<Integer> getParentIndices(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
    	Set<Integer> parents = new HashSet<Integer>();
    	for (IndexedBinaryRule<C, L> rule : indexedBinaryRules) {
    		parents.add(rule.parent().gpu());
    	}
    	return parents;
    }
    
    private List<IndexedBinaryRule<C, L>>[] balancedSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int numSegments) {
    	IndexingILPWrapper<String> ilp = new IndexingILPWrapper<String>(new CPLEXIntegerLinearProgram(1, false));
    	ilp.addBoundedVar("maxSegmentSize", 0, Integer.MAX_VALUE);
    	for (int segmentId=0; segmentId<numSegments; ++segmentId) {
    		for (int ruleId=0; ruleId<indexedBinaryRules.size(); ++ruleId) {
    			ilp.addBoundedIntVar("ruleAssignment"+segmentId+"_"+ruleId, 0, Integer.MAX_VALUE);
    		}
    	}
    	
    	List<IndexedBinaryRule<C, L>>[] result = new List[numSegments];
    	for (int i=0; i<numSegments; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	return result;
    }

}
