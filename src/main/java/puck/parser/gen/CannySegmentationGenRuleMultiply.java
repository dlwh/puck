package puck.parser.gen;

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
	
	private List<IndexedBinaryRule<C, L>>[] segmentBinariesMajor(List<IndexedBinaryRule<C, L>> indexedBinaryRules) {
		int numMajorSegments = BINARY_PARENT_NUM_MAJOR_SEGMENTS * BINARY_LEFT_NUM_MAJOR_SEGMENTS * BINARY_RIGHT_NUM_MAJOR_SEGMENTS;
		List<IndexedBinaryRule<C, L>>[] result = new List[numMajorSegments];
	}
}
