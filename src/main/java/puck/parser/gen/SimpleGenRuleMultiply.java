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
public class SimpleGenRuleMultiply<C, L> extends JavaFriendlyGenRuleMultiply<C, L> {

    private RuleStructure<C, L> structure;

    public SimpleGenRuleMultiply(RuleStructure<C, L> structure) {
        this.structure = structure;
    }

    @Override
    public CLBinaryRuleUpdater javaBinaryRuleApplication(List<IndexedBinaryRule<C, L>> indexedBinaryRules, String name, CLContext context) {
        ArrayList<String> kernelTexts = new ArrayList<String>();

//        if(4 + 4 == 8)
//            throw new UnsupportedOperationException("Not implemented!");

        // todo: partition the grammar

        kernelTexts.add(binaryKernelText(name, indexedBinaryRules));

        List<CLKernel> kernels = compileKernels(context, kernelTexts);
        int[] globalSize = {32 * 40, 1, 8};
        int[] wgSize = {32, 1, 1};

        return new CLBinaryRuleUpdater(kernels, globalSize, wgSize);
    }



    @Override
    public CLUnaryRuleUpdater javaUnaryRuleApplication(List<IndexedUnaryRule<C, L>> indexedUnaryRules, String name, CLContext context) {
        ArrayList<String> kernelTexts = new ArrayList<String>();

        kernelTexts.add(unaryKernelText(name, indexedUnaryRules));

        List<CLKernel> kernels = compileKernels(context, kernelTexts);

        return new CLUnaryRuleUpdater(kernels);
    }

    private String unaryKernelText(String name, List<IndexedUnaryRule<C, L>> partition) {
        // todo: subpartition the grammar
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
        for(IndexedUnaryRule<C, L> rule: partition) {
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

//        sb.append("      switch(grammarSubPartition) {\n" +
//              "        ${subpartitionsKernels.map {case  (id, (fnname, _)) => s\"case $id: $fnname(mask, parents, row, child, right, numRows); continue;\" }.mkString(\"\\n            \")}\n" +
//              "       default: continue;\n" +
//              "       }\n" +
//              "}");

        sb.append("    }\n");
        sb.append("}\n");
        return sb.toString();

    }

    private String binaryKernelText(String name, List<IndexedBinaryRule<C, L>> partition) {
        // todo: subpartition the grammar
        StringBuilder sb = new StringBuilder();


        sb.append(CLMaskKernels.maskHeader(structure));
        sb.append("\n\n");

        sb.append(String.format(
                " __kernel void %s(__global volatile float* parents, __global int* parentIndex, __global float* left, __global float* right, __global const mask_t* masks, int numRows, int cellsToDo) {\n" +
                "    int numWorkers = get_global_size(0);\n" +
                "    int grammarSubPartition = get_group_id(1);\n" +
                "    for (int row = get_global_id(0); row < cellsToDo; row += numWorkers) {\n" +
                "      const mask_t mask = masks[parentIndex[row]];\n", name));

        sb.append(String.format("    if (%s) continue;", CLMaskKernels.genCheckIfMaskIsEmpty(structure, "mask", getParents(partition))));

        sb.append("\n\n");

        Map<Integer,String> declaredParents = new HashMap<Integer, String>(),
                declaredLeft = new HashMap<Integer, String>(),
                declaredRight = new HashMap<Integer, String>();

        // todo: reorder to sensible groupings
        for(IndexedBinaryRule<C, L> rule: partition) {
            int parentIndex = rule.rule().parent().gpu();
            String parent = declaredParents.get(parentIndex);
            if(parent == null) {
                parent = "parent_" + parentIndex;
                sb.append(String.format("    float parent_%d = parents[%d * numRows + row];\n", parentIndex, parentIndex));
                declaredParents.put(parentIndex, parent);
            }

            int leftIndex = rule.rule().left().gpu();
            String left = declaredLeft.get(leftIndex);
            if(left == null) {
                left = "left_" + leftIndex;
                sb.append(String.format("    float left_%d = left[%d * numRows + row];\n", leftIndex, leftIndex));
                declaredLeft.put(leftIndex, left);
            }

            int rightIndex = rule.rule().right().gpu();
            String right = declaredRight.get(rightIndex);
            if(right == null) {
                right = "right_" + rightIndex;
                sb.append(String.format("    float right_%d = right[%d * numRows + row];\n", rightIndex, rightIndex));
                declaredRight.put(rightIndex, right);
            }

            sb.append(String.format("     %s = max(%s, %s + %s + %ff);\n", parent, parent, left, right, structure.scores()[rule.ruleId()]));
        }


        sb.append("     // write out\n");
        for(Map.Entry<Integer, String> e: declaredParents.entrySet()) {
            sb.append(String.format("    parents[%d * numRows + row] = %s;\n", e.getKey(), e.getValue()));
        }

//        sb.append("      switch(grammarSubPartition) {\n" +
//              "        ${subpartitionsKernels.map {case  (id, (fnname, _)) => s\"case $id: $fnname(mask, parents, row, left, right, numRows); continue;\" }.mkString(\"\\n            \")}\n" +
//              "       default: continue;\n" +
//              "       }\n" +
//              "}");

        sb.append("    }\n");
        sb.append("}\n");

        return sb.toString();

    }



}
