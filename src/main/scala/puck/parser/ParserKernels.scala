package puck.parser

import trochee.kernels.{KernelOps, KernelOpsExp}
import virtualization.lms.common._
import reflect.SourceContext
import trochee.codegen._
import virtualization.lms.internal.{Expressions, FatExpressions, Effects}
import virtualization.lms.common.RangeOpsExp
import virtualization.lms.common.RangeOps
import virtualization.lms.common.Base
import trochee.basic.{SpireOpsExp, GenSpireOps}

/**
 * 
 * @author dlwh
 */
abstract class ParserKernels extends InsideKernels with KernelOps with RangeOps { this: Base =>


}

object NullGrammar extends ParserKernels with InliningInsideKernels with ParserCommonExp with KernelOpsExp with RangeOpsExp with IfThenElseExp with SpireOpsExp { self =>
  val codegen = new OpenCLKernelCodegen with OpenCLKernelGenArrayOps with OpenCLParserGen {
    val IR: self.type = self
  }

  type Real = Float

  def manifestReal: Manifest[Real] = manifest[Float]

  def rigRepReal: Numeric[NullGrammar.Real] = implicitly


  val grammar = Grammar.parseFile(new java.io.File("src/main/resources/trochee/parser/demo.grammar.txt"))
  def numGrammars: Int = 1

  def main(args: Array[String]) {
    println(codegen.mkKernel(insideUnaries))
  }
  register()
}


