package puck.parser.gen;

import java.util.ArrayList;
import java.util.List;

import puck.parser.RuleStructure;

public class CannySegmentationGenRuleMultiply<C, L>  extends SimpleGenRuleMultiply<C, L> {
	
	public static final int UNARY_PARENT_NUM_MAJOR_SEGMENTS = 1;
	public static final int UNARY_CHILD_NUM_MAJOR_SEGMENTS = 1;
	
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
		List<IndexedUnaryRule<C, L>>[] segmentation = squareSegmentUnaries(indexedUnaryRules, UNARY_PARENT_NUM_MAJOR_SEGMENTS, UNARY_CHILD_NUM_MAJOR_SEGMENTS);
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
		List<IndexedBinaryRule<C, L>>[] segmentation = cubeSegmentBinaries(indexedBinaryRules, BINARY_PARENT_NUM_MAJOR_SEGMENTS, BINARY_LEFT_NUM_MAJOR_SEGMENTS, BINARY_RIGHT_NUM_MAJOR_SEGMENTS);
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for (List segment : segmentation) {
			min = Math.min(segment.size(), min);
			max = Math.max(segment.size(), max);
		}
		System.out.println("min binary segment size: "+min);
		System.out.println("max binary segment size: "+max);
		List<IndexedBinaryRule<C, L>>[][] subsegmentation = new List[segmentation.length][];
		for (int i=0; i<segmentation.length; ++i) {
			subsegmentation[i] = cubeSegmentBinaries(segmentation[i], BINARY_PARENT_NUM_MINOR_SEGMENTS, BINARY_LEFT_NUM_MINOR_SEGMENTS, BINARY_RIGHT_NUM_MINOR_SEGMENTS);
		}
		return subsegmentation;
	}
	
	private List<IndexedBinaryRule<C, L>>[] cubeSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int parentNumSeg, int leftNumSeg, int rightNumSeg) {
		int ruleNumSeg = parentNumSeg * leftNumSeg * rightNumSeg;
		List<IndexedBinaryRule<C, L>>[] result = new List[ruleNumSeg];
		for (int i=0; i<ruleNumSeg; i++) {
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
	
	private List<IndexedUnaryRule<C, L>>[] squareSegmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules, int parentNumSeg, int childNumSeg) {
		int ruleNumSeg = parentNumSeg * childNumSeg;
		List<IndexedUnaryRule<C, L>>[] result = new List[ruleNumSeg];
		for (int i=0; i<ruleNumSeg; i++) {
			result[i] = new ArrayList<IndexedUnaryRule<C, L>>();
		}
		for (IndexedUnaryRule<C, L> rule : indexedUnaryRules) {
			int parentSegment = rule.rule().parent().gpu() % parentNumSeg;
			int childSegment = rule.rule().child().gpu() % childNumSeg;
			int ruleSegment = parentSegment * (childNumSeg) + childSegment; 
			result[ruleSegment].add(rule);
		}
		return result;
	}
}
