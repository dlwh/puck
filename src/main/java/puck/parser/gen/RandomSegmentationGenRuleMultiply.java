package puck.parser.gen;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import puck.parser.RuleStructure;

public class RandomSegmentationGenRuleMultiply<C, L> extends SimpleGenRuleMultiply<C, L> {

	public static final int BINARY_SEGMENT_SIZE = 2000;
	
	public RandomSegmentationGenRuleMultiply(RuleStructure<C, L> structure) {
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
		List<IndexedBinaryRule<C, L>>[] segmentation = randomSegmentation(indexedBinaryRules, BINARY_SEGMENT_SIZE, new Random(1));
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
			subsegmentation[i] = modSegmentBinaries(indexedBinaryRules, NUM_SM);
		}
		return subsegmentation;
	}

    private List<IndexedBinaryRule<C, L>>[] modSegmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules, int m) {
    	List<IndexedBinaryRule<C, L>>[] result = new List[m];
    	for (int i=0; i<m; ++i) result[i] = new ArrayList<IndexedBinaryRule<C, L>>();
    	for(IndexedBinaryRule<C, L> rule: indexedBinaryRules) {
    		result[rule.rule().parent().gpu() % m].add(rule);
    	}
    	return result;
    }
	
    private static <T> List<T>[] randomSegmentation(List<T> list, int segSize, Random rand) {
    	List<T> shuffledList = new ArrayList<T>(list);
    	Collections.shuffle(shuffledList, rand);
    	int numSegs = (int) Math.ceil(((double) shuffledList.size()) / segSize);
    	List<T>[] result = new List[numSegs];
        for (int s=0; s<numSegs; ++s) {
        	result[s] = shuffledList.subList(s*segSize, Math.min(shuffledList.size(), (s+1)*segSize));
        }
    	return result;
    }
    
}
