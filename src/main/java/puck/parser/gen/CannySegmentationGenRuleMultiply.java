package puck.parser.gen;

import java.util.ArrayList;
import java.util.List;

import puck.parser.RuleStructure;

public class CannySegmentationGenRuleMultiply<C, L>  extends SimpleGenRuleMultiply<C, L> {
	
	public static final int BINARY_PARENT_NUM_MAJOR_SEGMENTS = 6;
	public static final int BINARY_LEFT_NUM_MAJOR_SEGMENTS = 2;
	public static final int BINARY_RIGHT_NUM_MAJOR_SEGMENTS = 2;
	
	public static final int BINARY_PARENT_NUM_MINOR_SEGMENTS = 2;
	public static final int BINARY_LEFT_NUM_MINOR_SEGMENTS = 2;
	public static final int BINARY_RIGHT_NUM_MINOR_SEGMENTS = 2;
	
	public CannySegmentationGenRuleMultiply(RuleStructure<C, L> structure) {
		super(structure);
	}

	public List<IndexedUnaryRule<C, L>>[] segmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules) {
		return null;
	}

	public List<IndexedBinaryRule<C, L>>[][] segmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
		return null;
	}
	
	private List<IndexedBinaryRule<C, L>>[] cubeSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int parentNumSeg, int leftNumSeg, int rightNumSeg) {
		int numMajorSegments = parentNumSeg * leftNumSeg * rightNumSeg;
		List<IndexedBinaryRule<C, L>>[] result = new List[numMajorSegments];
		for (int i=0; i<numMajorSegments; i++) {
			result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
		}
		for (IndexedBinaryRule<C, L> rule : indexedBinaryRules) {
			int parentSegment = rule.rule().parent().gpu() % parentNumSeg;
			int leftSegment = rule.rule().left().gpu() % leftNumSeg;
			int rightSegment = rule.rule().right().gpu() % rightNumSeg;
			int ruleSegment = parentSegment * (leftNumSeg*rightNumSeg) + leftSegment * rightNumSeg + rightSegment; 
			result[ruleSegment].add(rule);
		}
		return result;
	}
}
