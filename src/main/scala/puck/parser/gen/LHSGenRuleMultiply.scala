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

  case class Variable(name: String, descr: String) {

    def declare = s"float $name = ${floatToString(semiring.zero)}; // $descr"

    def repr = name
  }

  private def floatToString(x: Float) = x match {
    case Float.PositiveInfinity => "INFINITY"
    case Float.NegativeInfinity => "-INFINITY"
    case x if x.isNaN => "NAN"
    case x => s"${x}f"
  }

  def writeParent = {
    """ typedef union { int old; float oldf; } intbox;
      |
      |inline void write_parent(volatile __global float* loc, float value) {
      |  intbox old;
      |  old.oldf = value;
      |
      |  // TODO if not idempotent, have to do operator here.
      |  //while((old.old = atomic_cmpxchg((volatile __global int*)loc, old.old, *(int*)&value)) !=  *(int*)&value) value = max(value, old.oldf);
      |  *loc = value;
      |}
    """.stripMargin

  }

  def binaryRuleApplication(rulePartition: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLKernel = {
    val parents = rulePartition.map(_._1.parent).toSet
    val parentVariables: Map[Int, Variable] = parents.iterator.map(p => p.gpu -> Variable(s"parent_${p.gpu}", p.fineSym.toString)).toMap

    // set up the mask
    val maskStrings = for {
      (field, parentsInField) <- parents
                                .map(s => structure.refinements.labels.project(s.system))
                                .groupBy(_ / 32)
    } yield parentsInField.map(p => s"(1<<($p%32))").mkString(s"mask.fields[$field] & (","|",")")

    val checkMaskString = maskStrings.mkString("if (!((", ") | (", ")) ) return;")

      val text = structure.maskHeader + s"""
      $writeParent
    __kernel void $name(__global float* parents, __global int* parentIndex, __global float* left, __global float* right, __global float* ruleScores, __global const mask_t* masks, int numRows, int cellsToDo) {
        int row = get_global_id(0);
        if(row < cellsToDo) {
          const mask_t mask = masks[parentIndex[row]];
          $checkMaskString
          ${parentVariables.values.map(_.declare).mkString("\n        ")}
          ${coreRuleLoop(rulePartition, parentVariables)}
          ${{for( (id,v) <- parentVariables) yield s"write_parent(parents + numRows * $id + row, ${v.repr});"}.mkString("\n        ")}
        }
    }

    """
    val prog = cl.createProgram(text)
//    prog.addBuildOption("-cl-nv-verbose")
    prog.build().createKernels().head
  }

  private def coreRuleLoop(rulePartition: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], accumulator: Map[Int, Variable])(implicit cl: CLContext) = {
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
          sb ++= s"            ${accumulator(r.parent.gpu).repr} = ${semiring.mad(accumulator(r.parent.gpu).repr, jointName, s"ruleScores[$id]")};\n"
        }
      }
      sb ++= "         }\n"
    }
    sb.result()

  }

  def unaryRuleApplication(rulePartition: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLKernel = {
    val parents = rulePartition.map(_._1.parent).toSet
    val parentVariables = parents.iterator.map(p => p.gpu -> Variable(s"parent_${p.gpu}", p.fineSym.toString)).toMap
    val text = s"""
    __kernel void $name(__global float* parents, __global float* children, __global float* ruleScores, int numRows, int cellsToDo) {
        int row = get_global_id(0);
        if(row < cellsToDo) {
          ${parentVariables.values.map(_.declare).mkString("\n        ")}
          ${coreUnaryRuleLoop(rulePartition, parentVariables)}
          ${{for( (id,v) <- parentVariables) yield s"parents[numRows * $id + row] = ${v.repr};"}.mkString("\n        ")}
        }
    }

    """
    cl.createProgram(text).build().createKernels.head
  }

  private def coreUnaryRuleLoop(rulePartition: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], accumulator: Map[Int, Variable])(implicit cl: CLContext) = {
    val sb = new StringBuilder()
    for ((_lc, rr) <- rulePartition.groupBy(_._1.child)) {
      val lc = _lc.gpu
      val child = s"child_$lc"
      sb ++= s"        float $child = children[numRows * $lc + row];\n"
      for ((r, id) <- rr) {
        sb ++= s"            ${accumulator(r.parent.gpu).repr} = ${semiring.mad(accumulator(r.parent.gpu).repr, child, s"ruleScores[$id]")};\n"
      }
    }
    sb.result()
  }



}
