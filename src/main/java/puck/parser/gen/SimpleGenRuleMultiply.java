package puck.parser.gen;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLKernel;

import puck.parser.CLBinaryRuleUpdater;
import puck.parser.CLUnaryRuleUpdater;
import puck.parser.RuleStructure;
import puck.parser.SymId;

import java.util.*;

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
        boolean supportsExtendedAtomics =  supportsExtendedAtomics(context);
        for (int s=0; s<segments.length; s++) {
        	kernelTexts.add(binaryKernelText(name+s, segments[s], supportsExtendedAtomics));
        }

        List<CLKernel> kernels = compileKernels(context, kernelTexts);
        int[] globalSize = {WARP_SIZE * NUM_WARPS, NUM_SM, 1};
        int[] wgSize = {WARP_SIZE, 1, 1};
        return new CLBinaryRuleUpdater(kernels, globalSize, wgSize);
    }


    private String binaryKernelText(String name, List<IndexedBinaryRule<C, L>>[] subsegments, boolean supportsExtendedAtomics) {
        StringBuilder sb = new StringBuilder();
        sb.append(WRITE_PARENT_ATOMIC);
        sb.append(CLMaskKernels.maskHeader(structure));

        sb.append("\n\n");

        // Sort so that LHS priority, then RHS
        for(List<IndexedBinaryRule<C, L>> rules : subsegments) {
            Collections.sort(rules, new Comparator<IndexedBinaryRule<C, L>>() {
                @Override
                public int compare(IndexedBinaryRule<C, L> o1, IndexedBinaryRule<C, L> o2) {
                    int lhs = Integer.compare(o1.rule().left().gpu(), o2.rule().left().gpu());
                    if(lhs != 0) return lhs;
                    int rhs = Integer.compare(o1.rule().right().gpu(), o2.rule().right().gpu());
                    if(rhs != 0) return rhs;

                    int parent = Integer.compare(o1.rule().parent().gpu(), o2.rule().parent().gpu());

                    return parent;
                }
            });

        }

        Set<Integer> allParents = new HashSet<Integer>();
        Set<Integer> dupParents = new HashSet<Integer>();
        for(int m = 0; m < NUM_SM; ++m) {
            for(SymId<C, L> sym: getParents(subsegments[m])) {
                if(allParents.contains(sym.gpu())) {
                    dupParents.add(sym.gpu());
                } else {
                    allParents.add(sym.gpu());
                }
            }
        }

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
//        		sb.append(String.format("parents[%d * numRows + row] = %s;\n", e.getKey(), e.getValue()));
                String dest = String.format("parents[%d * numRows + row]", e.getKey());
                String src = e.getValue();
                sb.append(genWriteSymbol(dest, src, !dupParents.contains(e.getKey()), supportsExtendedAtomics));
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

    public static boolean GRAMMAR_IS_GENERATIVE = true;

    public String genWriteSymbol(String dest, String src, boolean symIsUniqueToSubsegmentation, boolean supportsExtendedAtomics) {
        if(false && symIsUniqueToSubsegmentation) {
            return String.format("%s = %s;\n", dest, src);
        } else if(GRAMMAR_IS_GENERATIVE && supportsExtendedAtomics) {
            return String.format("write_parent_gen_atomic(&%s, %s);\n", dest, src);
        } else {
            return String.format("write_parent_atomic(&%s, %s);\n", dest, src);
        }

    }

    // floats < 0 are well ordered such that if max(float1, float2) = float1, then min(*(int*)&float1,*(int*)&float2) = *(int*)&float1
    // note inversion of min and max
    // this is for write_atomic_min (because all floats are same sign)
    // there's a problem if one float is 0.0, but eh.
    private static final String WRITE_PARENT_ATOMIC = "" +
            "     typedef union { int old; float oldf; } intbox;\n" +
            "     \n" +
            "     inline void write_parent_atomic_gen(volatile __global float* loc, float value) {\n" +
            "       int newValue = atomic_min((volatile __global int*)loc, *(int*)&value);\n" +
            "       printf(\"%f %f %f\\n\", *loc, value, *(float*)&newValue);\n"+
            "      }\n"+
            "     \n" +
            "     inline void write_parent_atomic(volatile __global float* loc, float value) {\n" +
            "       intbox old;\n" +
            "       value = max(*loc, value);\n" +
            "       old.oldf = value;\n" +
            "     \n" +
            "       while((old.old = atomic_cmpxchg((volatile __global int*)loc, old.old, *(int*)&value)) !=  *(int*)&value) value = max(value, old.oldf);\n" +
            "     }\n\n\n";

}
