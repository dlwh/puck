package puck.parser.gen

import epic.trees.{UnaryRule, BinaryRule}
import com.nativelibs4java.opencl.{CLContext, CLKernel}
import puck.parser.SymId

/**
 * TODO
 *
 * @author dlwh
 **/
trait GenRuleMultiply {
  def binaryRuleApplication[C, L](rulePartition: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLKernel
  def unaryRuleApplication[C, L](rulePartition: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLKernel
}
