package puck.parser.gen

import epic.trees.{UnaryRule, BinaryRule}
import com.nativelibs4java.opencl.{CLKernel, CLContext}
import puck.parser.{RuleStructure, SymId, RuleSemiring}

/**
 * TODO
 *
 * @author dlwh
 **/
class LHSGenRuleMultiply[C, L](structure: RuleStructure[C, L])(implicit semiring: RuleSemiring) extends GenRuleMultiply[C, L] {
  def binaryRuleApplication(rulePartition: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLKernel = {
    val parents = rulePartition.map(_._1.parent).toSet
    val accumulator = semiring.accumulator(parents.map(_.gpu))
    // set up the mask
    val maskStrings = for {
//      field <- 0 to 3
      (field, parentsInField) <- parents
                                .map(s => structure.refinements.labels.project(s.system))
                                .groupBy(_ / 32)
    } yield parentsInField.map(p => s"(1<<($p%32))").mkString(s"mask.fields[$field] & (","|",")")
    val checkMaskString = maskStrings.mkString("if (!((", ") | (", ")) ) return;")
//    val checkMaskString = "if ( !any(( (__global int4*)masks)[row] )) return;"
//      val checkMaskString = "if ( !any( *( (int4*)&mask) )) return;"
      val text = structure.maskHeader + s"""
    __kernel void $name(__global float* parents, __global int* parentIndex, __global float* left, __global float* right, __global float* ruleScores, __global const mask_t* masks, int numRows, int cellsToDo) {
        int row = get_global_id(0);
        if(row < cellsToDo) {
          const mask_t mask = masks[parentIndex[row]];
//          if( mask.fields[0] != -1 || mask.fields[1] != -1 || mask.fields[2] != -1)
//            printf("%s %d %d %d qq %d %d${"\\n"}", __FUNCTION__, mask.fields[0], mask.fields[1], mask.fields[2], row, get_group_id(0));
          $checkMaskString
          ${accumulator.declare}
          ${coreRuleLoop(rulePartition, accumulator)}
          ${accumulator.output((id: Int) => s"parents[numRows * $id + row]")}
        }
    }

    """
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
