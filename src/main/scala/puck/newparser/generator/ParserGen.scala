package puck.newparser.generator

import trochee.codegen._
import scala.virtualization.lms.internal.{FatExpressions, Effects, Expressions}
import trochee.basic._
import trochee.kernels.KernelOpsExp
import scala.virtualization.lms.common.RangeOpsExp

/**
 * TODO
 *
 * @author dlwh
 **/
abstract class ParserGen[L] extends OpenCLKernelCodegen with OpenCLKernelGenArrayOps with OpenCLKernelGenVariables with GenSpireOps  with OpenCLKernelGenRangeOps {
//  val IR: scala.virtualization.lms.internal.Expressions with scala.virtualization.lms.internal.Effects with scala.virtualization.lms.internal.FatExpressions with trochee.kernels.KernelOpsExp with puck.parser.gen.ParserCommonExp[L] with scala.virtualization.lms.common.IfThenElseExp with trochee.basic.SpireOpsExp with puck.parser.gen.FloatOpsExp with RuleMultiply[L]
  val IR : Expressions with Effects with FatExpressions with ExtraBaseExp with KernelOpsExp with RangeOpsExp with RuleMultiply[L]
}
