package puck.parser.gen

import epic.trees.{UnaryRule, BinaryRule}
import com.nativelibs4java.opencl.{CLContext, CLKernel}

/**
 * TODO
 *
 * @author dlwh
 **/
trait GenRuleMultiply {
  def binaryRuleApplication(rulePartition: IndexedSeq[(BinaryRule[Int], Int)], name: String)(implicit cl: CLContext): CLKernel
  def unaryRuleApplication(rulePartition: IndexedSeq[(UnaryRule[Int], Int)], name: String)(implicit cl: CLContext): CLKernel
}
