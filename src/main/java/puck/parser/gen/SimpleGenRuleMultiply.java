package puck.parser.gen;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLKernel;

import puck.parser.CLBinaryRuleUpdater;
import puck.parser.CLUnaryRuleUpdater;
import puck.parser.RuleStructure;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * TODO
 *
 * @author dlwh
 */
public abstract class SimpleGenRuleMultiply<C, L> extends JavaFriendlyGenRuleMultiply<C, L> {
	
	public static final int WARP_SIZE = 32;
	public static final int NUM_WARPS = 48;
	public static final int NUM_SM = 8;
	
    private RuleStructure<C, L> structure;

    public SimpleGenRuleMultiply(RuleStructure<C, L> structure) {
        this.structure = structure;
    }
    
    public abstract List<IndexedUnaryRule<C, L>>[] segmentUnaries(List<IndexedUnaryRule<C, L>> indexedUnaryRules);

    public abstract List<IndexedBinaryRule<C, L>>[][] segmentBinaries(List<IndexedBinaryRule<C, L>> indexedBinaryRules);

    public CLBinaryRuleUpdater javaBinaryRuleApplication(List<IndexedBinaryRule<C, L>> indexedBinaryRules, String name, CLContext context) {
        ArrayList<String> kernelTexts = new ArrayList<String>();
        List<IndexedBinaryRule<C, L>>[][] segments = segmentBinaries(indexedBinaryRules);
        for (int s=0; s<segments.length; s++) {
        	kernelTexts.add(binaryKernelText(name+s, segments[s]));
        }
        List<CLKernel> kernels = compileKernels(context, kernelTexts);
        int[] globalSize = {WARP_SIZE * NUM_WARPS, NUM_SM, 1};
        int[] wgSize = {WARP_SIZE, 1, 1};
        return new CLBinaryRuleUpdater(kernels, globalSize, wgSize);
    }
    
    private String binaryKernelText(String name, List<IndexedBinaryRule<C, L>>[] subsegments) {
        StringBuilder sb = new StringBuilder();
        sb.append(CLMaskKernels.maskHeader(structure));
        sb.append("\n\n");
        
        for (int m=0; m<NUM_SM; ++m) {
        	sb.append("static void subpart"+m+"(const mask_t mask, __global volatile float* parents, int row, __global float* left, __global float* right, int numRows) {\n");
        	if (!subsegments[m].isEmpty()) sb.append(String.format("if (%s) return;\n", CLMaskKernels.genCheckIfMaskIsEmpty(structure, "mask", getParents(subsegments[m]))));        	
        	Map<Integer,String> declaredParents = new HashMap<Integer, String>();
        	Map<Integer,String> declaredLeft = new HashMap<Integer, String>();
        	Map<Integer,String>  declaredRight = new HashMap<Integer, String>();

        	// todo: reorder to sensible groupings
        	for(IndexedBinaryRule<C, L> rule : subsegments[m]) {
        		int parentIndex = rule.rule().parent().gpu();
        		String parent = declaredParents.get(parentIndex);
        		if(parent == null) {
        			parent = "parent_" + parentIndex;
        			sb.append(String.format("float parent_%d = parents[%d * numRows + row];\n", parentIndex, parentIndex));
        			declaredParents.put(parentIndex, parent);
        		}

        		int leftIndex = rule.rule().left().gpu();
        		String left = declaredLeft.get(leftIndex);
        		if(left == null) {
        			left = "left_" + leftIndex;
        			sb.append(String.format("float left_%d = left[%d * numRows + row];\n", leftIndex, leftIndex));
        			declaredLeft.put(leftIndex, left);
        		}

        		int rightIndex = rule.rule().right().gpu();
        		String right = declaredRight.get(rightIndex);
        		if(right == null) {
        			right = "right_" + rightIndex;
        			sb.append(String.format("float right_%d = right[%d * numRows + row];\n", rightIndex, rightIndex));
        			declaredRight.put(rightIndex, right);
        		}

        		sb.append(String.format("%s = max(%s, %s + %s + %ff);\n", parent, parent, left, right, structure.scores()[rule.ruleId()]));
        	}


        	sb.append("// write out\n");
        	for(Map.Entry<Integer, String> e: declaredParents.entrySet()) {
        		sb.append(String.format("parents[%d * numRows + row] = %s;\n", e.getKey(), e.getValue()));
        	}
        	sb.append("}\n\n");
        }

        sb.append(String.format(
                " __kernel void %s(__global volatile float* parents," +
                "                  __global int* parentIndex, " +
                "                  __global float* left, __global int* leftIndex, " +
                "                  __global float* right, __global int* rightIndex," +
                "                  __global const mask_t* masks, int numRows, int cellsToDo) {\n" +
                "    int numWorkers = get_global_size(0);\n" +
                "    int grammarSubPartition = get_group_id(1);\n" +
                "    for (int row = get_global_id(0); row < cellsToDo; row += numWorkers) {\n" +
                "      const mask_t mask = masks[parentIndex[row]];\n", name));
        sb.append("\n\n");
        
        sb.append("switch (grammarSubPartition) {\n");
        for (int m=0; m<NUM_SM; ++m) {
        	sb.append("case "+m+": subpart"+m+"(mask, parents, row, left, right, numRows); continue;\n");
        }
        sb.append("default: continue;\n");
        sb.append("}\n");

        sb.append("}\n");
        sb.append("}\n");
        return sb.toString();
    }

    public CLUnaryRuleUpdater javaUnaryRuleApplication(List<IndexedUnaryRule<C, L>> indexedUnaryRules, String name, CLContext context) {
        ArrayList<String> kernelTexts = new ArrayList<String>();
        List<IndexedUnaryRule<C, L>>[] segments = segmentUnaries(indexedUnaryRules);
        for (int s=0; s<segments.length; s++) {
        	kernelTexts.add(unaryKernelText(name+s, segments[s]));
        }
        List<CLKernel> kernels = compileKernels(context, kernelTexts);
        return new CLUnaryRuleUpdater(kernels);
    }

    private String unaryKernelText(String name, List<IndexedUnaryRule<C, L>> segment) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
                " __kernel void %s(__global volatile float* parents, __global float* child, int numRows, int cellsToDo) {\n" +
                "    int numWorkers = get_global_size(0);\n" +
                "    int grammarSubPartition = get_group_id(1);\n" +
                "    for (int row = get_global_id(0); row < cellsToDo; row += numWorkers) {\n", name));
        sb.append("\n\n");

        Map<Integer,String> declaredParents = new HashMap<Integer, String>(),
                declaredLeft = new HashMap<Integer, String>();

        // todo: reorder to sensible groupings
        for(IndexedUnaryRule<C, L> rule : segment) {
        	int parentIndex = rule.rule().parent().gpu();
        	String parent = declaredParents.get(parentIndex);
        	if(parent == null) {
        		parent = "parent_" + parentIndex;
        		sb.append(String.format("float parent_%d = parents[%d * numRows + row];\n", parentIndex, parentIndex));
        		declaredParents.put(parentIndex, parent);
        	}
        	
        	int childIndex = rule.rule().child().gpu();
        	String child = declaredLeft.get(childIndex);
        	if(child == null) {
        		child = "child_" + childIndex;
        		sb.append(String.format("float child_%d = child[%d * numRows + row];\n", childIndex, childIndex));
        		declaredLeft.put(childIndex, child);
        	}
        	
        	
        	sb.append(String.format("%s = max(%s, %s + %ff);\n", parent, parent, child, structure.scores()[rule.ruleId()]));
        }

        sb.append("// write out\n");
        for(Map.Entry<Integer, String> e: declaredParents.entrySet()) {
            sb.append(String.format("parents[%d * numRows + row] = %s;\n", e.getKey(), e.getValue()));
        }

        sb.append("}\n");
        sb.append("}\n");
        return sb.toString();
    }

}
