package puck.parser.gen;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import puck.parser.RuleSemiring;
import puck.parser.RuleStructure;

public class CannyGrammarClusterer<C, L> implements GrammarClusterer<C, L> {
	
	public static final int BINARY_PARENT_NUM_MAJOR_SEGMENTS = 6;
	public static final int BINARY_LEFT_NUM_MAJOR_SEGMENTS = 2;
	public static final int BINARY_RIGHT_NUM_MAJOR_SEGMENTS = 2;
	
	public static final int BINARY_PARENT_NUM_MINOR_SEGMENTS = 2;
	public static final int BINARY_LEFT_NUM_MINOR_SEGMENTS = 2;
	public static final int BINARY_RIGHT_NUM_MINOR_SEGMENTS = 2;

    private Logger logger = LoggerFactory.getLogger(this.getClass().getName());

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
		List<IndexedBinaryRule<C, L>>[] segmentation = cubeSubsegmentBinaries(indexedBinaryRules, BINARY_PARENT_NUM_MAJOR_SEGMENTS, BINARY_LEFT_NUM_MAJOR_SEGMENTS, BINARY_RIGHT_NUM_MAJOR_SEGMENTS);
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for (List segment : segmentation) {
			min = Math.min(segment.size(), min);
			max = Math.max(segment.size(), max);
		}
		logger.info("min binary segment size: "+min);
		logger.info("max binary segment size: "+max);
		List<IndexedBinaryRule<C, L>>[][] subsegmentation = new List[segmentation.length][];
		for (int i=0; i<segmentation.length; ++i) {
			subsegmentation[i] = cubeSubsegmentBinaries(segmentation[i], BINARY_PARENT_NUM_MINOR_SEGMENTS, BINARY_LEFT_NUM_MINOR_SEGMENTS, BINARY_RIGHT_NUM_MINOR_SEGMENTS);
//			subsegmentation[i] = modSubsegmentBinariesByParent(segmentation[i], NUM_SM);
		}
		return subsegmentation;
	}
	
	private List<IndexedBinaryRule<C, L>>[] cubeSubsegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int parentNumSeg, int leftNumSeg, int rightNumSeg) {
		int ruleNumSeg = parentNumSeg * leftNumSeg * rightNumSeg;
		int maxParent = 0;
		int maxLeft = 0;
		int maxRight = 0;
		for (IndexedBinaryRule<C, L> rule : indexedBinaryRules) {
			maxParent = Math.max(maxParent, rule.rule().parent().gpu());
			maxLeft = Math.max(maxLeft, rule.rule().left().gpu());
			maxRight = Math.max(maxRight, rule.rule().right().gpu());
		}
		int parentSegSize = (int) Math.ceil((maxParent + 1.0) / parentNumSeg);
		int leftSegSize = (int) Math.ceil((maxLeft + 1.0) / leftNumSeg);
		int rightSegSize = (int) Math.ceil((maxRight + 1.0) / rightNumSeg);
		List<IndexedBinaryRule<C, L>>[] result = new List[ruleNumSeg];
		for (int i=0; i<ruleNumSeg; i++) {
			result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
		}
		for (IndexedBinaryRule<C, L> rule : indexedBinaryRules) {
			int parentSegment = rule.rule().parent().gpu() / parentSegSize;
			int leftSegment = rule.rule().left().gpu() / leftSegSize;
			int rightSegment = rule.rule().right().gpu() / rightSegSize;
			int ruleSegment = parentSegment * (leftNumSeg*rightNumSeg) + leftSegment * rightNumSeg + rightSegment; 
			result[ruleSegment].add(rule);
		}
		return result;
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
