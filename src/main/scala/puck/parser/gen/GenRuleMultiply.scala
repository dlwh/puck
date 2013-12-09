package puck.parser.gen

import epic.trees.{UnaryRule, BinaryRule}
import com.nativelibs4java.opencl.{CLContext, CLKernel}
import puck.parser.SymId

/**
 * TODO
 *
 * @author dlwh
 **/
trait GenRuleMultiply[C, L] {
  def binaryRuleApplication(rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLBinaryRuleUpdater
  def unaryRuleApplication(rulePartition: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLKernel
}
