package puck.parser.gen;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLKernel;

import puck.package$;
import puck.parser.*;

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
	
    public RuleStructure<C, L> structure;
    private boolean writeDirectToChart;

    public SimpleGenRuleMultiply(RuleStructure<C, L> structure, boolean writeDirectToChart) {
        super(structure, writeDirectToChart);
        this.structure = structure;
        this.writeDirectToChart = writeDirectToChart;
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

        List<RuleKernel> kernels = compileKernels(context, this.<IndexedBinaryRule<C, L>>flatten(segments), kernelTexts);
        int[] globalSize = {WARP_SIZE * NUM_WARPS, NUM_SM, 1};
        int[] wgSize = {WARP_SIZE, 1, 1};
        return new CLBinaryRuleUpdater(kernels, globalSize, wgSize, writeDirectToChart);
    }


    private String binaryKernelText(String name, List<IndexedBinaryRule<C, L>>[] subsegments, boolean supportsExtendedAtomics) {
      StringBuilder sb = new StringBuilder();

      // determine duplicate parents
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


        if(!dupParents.isEmpty() && supportsExtendedAtomics) {
            sb.append("#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n");
        }

        sb.append(WRITE_PARENT_ATOMIC);
        sb.append(CLMaskKernels.maskHeader(structure));

        sb.append("\n\n");

        // Sort so that LHS priority, then RHS
        for(List<IndexedBinaryRule<C, L>> rules : subsegments) {
            Collections.sort(rules, new Comparator<IndexedBinaryRule<C, L>>() {
                @Override
                public int compare(IndexedBinaryRule<C, L> o1, IndexedBinaryRule<C, L> o2) {
    				int parent = Integer.compare(o1.rule().parent().gpu(), o2.rule().parent().gpu());
    				if(parent != 0) return parent;
    				int lhs = Integer.compare(o1.rule().left().gpu(), o2.rule().left().gpu());
    				if(lhs != 0) return lhs;
    				int rhs = Integer.compare(o1.rule().right().gpu(), o2.rule().right().gpu());
    				return rhs;
                }
            });

        }

        for (int m=0; m<NUM_SM; ++m) {
        	sb.append("static void subpart"+m+"(const mask_t mask, __global volatile float* parents, __global int* parentIndex, int row, __global float* left, __global float* right, int numRows) {\n");
//        	if (!subsegments[m].isEmpty()) sb.append(String.format("if (%s) return;\n", CLMaskKernels.genCheckIfMaskIsEmpty(structure, "mask", getParents(subsegments[m]))));
        	Map<Integer,String> declaredParents = new HashMap<Integer, String>();
        	Map<Integer,String> declaredLeft = new HashMap<Integer, String>();
        	Map<Integer,String> declaredRight = new HashMap<Integer, String>();

            sb.append("int pi = parentIndex[row];");

        	Map<Integer,Integer> parentCounts = new HashMap<Integer,Integer>();
        	for(IndexedBinaryRule<C, L> rule : subsegments[m]) { 
        		int parentIndex = rule.rule().parent().gpu();
        		Integer count = parentCounts.get(parentIndex);
        		if (count == null) {
        			count = 0;
        		}
        		count++;
        		parentCounts.put(parentIndex, count);
        	}
        	
        	int cellSize = package$.MODULE$.roundUpToMultipleOf(Math.max(structure.numNonTerms(), structure.numTerms()), 32);
        	for(IndexedBinaryRule<C, L> rule : subsegments[m]) {
        		int parentIndex = rule.rule().parent().gpu();
        		String parent = declaredParents.get(parentIndex);
        		if(parent == null) {
        			parent = "parent_" + parentIndex;
                    if(writeDirectToChart)
                        sb.append(String.format("float parent_%d = parents[pi * "+cellSize+" + %d];\n", parentIndex, parentIndex));
        			else
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
        		
        		parentCounts.put(parentIndex, parentCounts.get(parentIndex)-1);
        		if (parentCounts.get(parentIndex) == 0) {

                    if(writeDirectToChart) {
                        String dest = String.format("parents[pi * "+cellSize+" + %d]", parentIndex);
                        String src = parent;
                        sb.append(genWriteSymbol(dest, src, false, supportsExtendedAtomics));
                    } else {
                        String dest = String.format("parents[%d * numRows + row]", parentIndex);
                        String src = parent;
                        sb.append(genWriteSymbol(dest, src, !dupParents.contains(parentIndex), supportsExtendedAtomics));
                    }

        		}
        	}


//        	sb.append("// write out\n");
//        	for(Map.Entry<Integer, String> e: declaredParents.entrySet()) {
//        		sb.append(String.format("parents[%d * numRows + row] = %s;\n", e.getKey(), e.getValue()));
//                String dest = String.format("parents[%d * numRows + row]", e.getKey());
//                String src = e.getValue();
//                sb.append(genWriteSymbol(dest, src, !dupParents.contains(e.getKey()), supportsExtendedAtomics));
//        	}
        	sb.append("}\n\n");
        }

        sb.append(String.format(
                " __kernel void %s(__global volatile float* parents," +
                "                  __global int* parentIndex, " +
                "                  __global float* left," +
                "                  __global float* right,"  +
                "                  __global const mask_t* masks, int numRows, int cellsToDo) {\n" +
                "    int numWorkers = get_global_size(0);\n" +
                "    int grammarSubPartition = get_group_id(1);\n" +
                "    for (int row = get_global_id(0); row < cellsToDo; row += numWorkers) {\n" +
                "      const mask_t mask = masks[parentIndex[row]];\n", name));
        sb.append("\n\n");
        
        sb.append("switch (grammarSubPartition) {\n");
        for (int m=0; m<NUM_SM; ++m) {
        	sb.append("case "+m+": subpart"+m+"(mask, parents, parentIndex, row, left, right, numRows); continue;\n");
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
        List<RuleKernel> kernels = compileKernels(context, Arrays.asList(segments), kernelTexts);
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

    public static boolean GRAMMAR_IS_GENERATIVE = false;
    public static boolean NVIDIA_IS_STILL_STUPID = true;

    public String genWriteSymbol(String dest, String src, boolean symIsUniqueToSubsegmentation, boolean supportsExtendedAtomics) {
        if(symIsUniqueToSubsegmentation) {
            return String.format("%s = %s;\n", dest, src);
        } else if(GRAMMAR_IS_GENERATIVE && supportsExtendedAtomics && NVIDIA_IS_STILL_STUPID) {
            return String.format("write_parent_atomic_nvidia_gen(&%s, %s);\n", dest, src);
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
            "     inline void write_parent_gen_atomic(volatile __global float* loc, float value) {\n" +
            "        atomic_min((volatile __global int*)loc, *(int*)&value);\n" +
            "      }\n"+
            " #ifdef NVIDIA \n" +
            "     inline void write_parent_atomic_nvidia_gen(volatile __global float* loc, float value) {\n" +
            "        volatile __global int* d_ptr = (volatile __global int*)loc;\n" +
            "        int z = *(int*)&value;\n" +
            "        asm(\"atom.global.max.s32 %%r8, [%0], %1;\" :: \"l\"(d_ptr), \"r\"(z));\n" +
            "      }\n"+
            "     \n" +
            " #endif \n" +
            "     inline void write_parent_atomic(volatile __global float* loc, float value) {\n" +
            "       intbox old;\n" +
            "       value = max(*loc, value);\n" +
            "       old.oldf = value;\n" +
            "     \n" +
            "       while((old.old = atomic_cmpxchg((volatile __global int*)loc, old.old, *(int*)&value)) !=  *(int*)&value) value = max(value, old.oldf);\n" +
            "     }\n\n\n";

}
