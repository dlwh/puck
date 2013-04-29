package puck.parser.gen

import trochee.kernels.{KernelOps, KernelOpsExp}
import virtualization.lms.common._
import trochee.codegen._
import virtualization.lms.common.RangeOpsExp
import virtualization.lms.common.RangeOps
import virtualization.lms.common.Base
import trochee.basic.SpireOpsExp
import puck.parser.{GPUCharts, RuleStructure}
import scala.Float
import scala.collection.mutable.ArrayBuffer

/**
 * 
 * @author dlwh
 */
abstract class ParserKernels[L] extends InsideKernels[L] with KernelOps with RangeOps { this: Base =>


}

abstract class ParserGenerator[L](val grammar: RuleStructure[_, L], val numGrammars: Int = 1) extends ParserKernels[L] with InliningInsideKernels[L] with InliningOutsideKernels[L] with ParserCommonExp[L] with KernelOpsExp with RangeOpsExp with IfThenElseExp with SpireOpsExp with FloatOpsExp { self =>
  val codegen = new OpenCLKernelCodegen with OpenCLKernelGenArrayOps with OpenCLParserGen[L] with OpenCLKernelGenRangeOps {
    val IR: self.type = self
  }

  register()
}

