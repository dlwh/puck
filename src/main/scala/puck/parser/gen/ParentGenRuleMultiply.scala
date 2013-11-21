package puck.parser.gen

import epic.trees.{UnaryRule, BinaryRule}
import com.nativelibs4java.opencl.{CLKernel, CLContext}
import puck.parser.{RuleStructure, SymId, RuleSemiring}

/**
 * TODO
 *
 * @author dlwh
 **/
class ParentGenRuleMultiply[C, L](structure: RuleStructure[C, L])(implicit semiring: RuleSemiring) extends GenRuleMultiply[C, L] {
  def binaryRuleApplication(rulePartition: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLKernel = {

    val byCoarseParents = rulePartition.groupBy(r => structure.refinements.labels.project(r._1.parent.system))

    var text = structure.maskHeader + s"""
    __kernel void $name(__global float* parents, __global float* left, __global float* right, __global float* ruleScores, __constant const mask_t* masks, int numRows, int cellsToDo) {
        int row = get_global_id(0);
        if(row < cellsToDo) {
    """
    for((coarseParent, rulePartition) <- byCoarseParents) {
      val parents = rulePartition.map(_._1.parent).toSet
      val accumulator = semiring.accumulator(parents.map(_.gpu))
      // TODO: don't copy/paste
      text +=  s"""
      if ((masks[row].fields[$coarseParent/32] & ($coarseParent%32)) != 0) {
          ${accumulator.declare}
          ${coreRuleLoop(rulePartition, accumulator)}
          ${accumulator.output((id: Int) => s"parents[numRows * $id + row]")}


       }"""
    }

    text += """
      return;
        }

    }

            """

    println(text)
    cl.createProgram(text).build().createKernels().head
  }

  private def coreRuleLoop(rulePartition: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], accumulator: semiring.Accumulator)(implicit cl: CLContext) = {
    val sb = new StringBuilder()
    for ((_lc, rr) <- rulePartition.groupBy(_._1.left)) {
      val lc = _lc.gpu
      sb ++= s"        float leftChild_$lc = left[numRows * $lc + row];\n"
//      sb ++= s"        if (leftChild_$lc > 0) printf("+'"' + s"LC WTF $lc %s %f" +"\\n\"" + s", __FUNCTION__, leftChild_$lc);\n"
      sb ++= s"        {\n"
      for ((_rc, rrr) <- rr.groupBy(_._1.right)) {
        val rc = _rc.gpu
        sb ++= s"            float rightChild_$rc = right[numRows * $rc + row];\n"
//        sb ++= s"            if (rightChild_$rc > 0) printf("+'"' + s"RC WTF $rc %s %f" +"\\n\"" + s", __FUNCTION__, rightChild_$rc);\n"
        val jointName = s"joint_${lc}_${rc}"
        sb ++= s"            float $jointName = ${semiring.times(s"leftChild_$lc", s"rightChild_$rc")};\n"
        for ((r, id) <- rrr) {
          sb ++= s"            ${accumulator.mad(r.parent.gpu, jointName, s"ruleScores[$id]")};\n"
        }
      }
      sb ++= "         }\n"
    }
    sb.result()

  }

  def unaryRuleApplication(rulePartition: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLKernel = {
    val parents = rulePartition.map(_._1.parent).toSet
    val accumulator = semiring.accumulator(parents.map(_.gpu))
    val text = s"""
    __kernel void $name(__global float* parents, __global float* children, __global float* ruleScores, int numRows, int cellsToDo) {
        int row = get_global_id(0);
        if(row < cellsToDo) {
          ${accumulator.declare}
          ${coreUnaryRuleLoop(rulePartition, accumulator)}
          ${accumulator.output((id: Int) => s"parents[numRows * $id + row]")}
        }
    }

    """
    cl.createProgram(text).build().createKernels.head
  }

  private def coreUnaryRuleLoop(rulePartition: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], accumulator: semiring.Accumulator)(implicit cl: CLContext) = {
    val sb = new StringBuilder()
    for ((_lc, rr) <- rulePartition.groupBy(_._1.child)) {
      val lc = _lc.gpu
      val child = s"child_$lc"
      sb ++= s"        float $child = children[numRows * $lc + row];\n"
      for ((r, id) <- rr) {
        sb ++= s"          ${accumulator.mad(r.parent.gpu, child, s"ruleScores[$id]")};\n"
      }
    }
    sb.result()
  }



}
