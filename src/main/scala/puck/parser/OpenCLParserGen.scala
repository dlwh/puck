package puck.parser

import scala.virtualization.lms.internal.{FatExpressions, Effects, Expressions}
import trochee.codegen.OpenCLKernelCodegen
import trochee.basic.{SpireOpsExp, GenSpireOps}
import scala.virtualization.lms.common.IfThenElseExp

trait OpenCLParserGen extends OpenCLKernelCodegen with GenSpireOps {
  val IR: Expressions with Effects with FatExpressions with trochee.kernels.KernelOpsExp with ParserCommonExp with IfThenElseExp with SpireOpsExp
  import IR._
  lazy val typeMaps = {
    Map[Class[_],String](manifestParseChart.erasure -> "PARSE_CELL" ,
    manifestTermChart.erasure -> "PARSE_CELL",
    manifestRuleCell.erasure -> "rule_cell*")
}
  override def remap[A](m: Manifest[A]) : String = {
    typeMaps.getOrElse(m.erasure, super.remap(m))
  }

  override def quote(x: Exp[Any]) = x match {
    case RuleDeref(cell, rule, grammar) => preferNoLocal(cell) + s"->rules[${quote(rule)}][${quote(grammar)}]"
    case _ => super.quote(x)
  }

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) {
    rhs match {
      case Mad(a,b,c) =>
        cacheAndEmit(sym, s"mad(${quote(a)}, ${quote(b)}, ${quote(c)})")
      case app@CellApply(NTCell(cell, off, begin, end, gram), symsym) =>
        cacheAndEmit(addPos(sym, app), s"${quote(cell)}[(${quote(symsym)} * CHART_SIZE + ${quote(off)} + TRIANGULAR_INDEX(${quote(begin)}, ${quote(end)}))*NUM_GRAMMARS + ${quote(gram)}]")
      case app@CellApply(TCell(cell, off, begin, gram), symsym) =>
        cacheAndEmit(addPos(sym, app), s"${quote(cell)}[(${quote(symsym)} * CHART_SIZE + ${quote(off)} + ${quote(begin)})*NUM_GRAMMARS + ${quote(gram)}]")
      case MadUpdate(acc, index, a, b) =>
        val id = acc.prefix + index
        if(!acc.declared(index)) {
          acc.declared += index
          cacheAndEmit(sym, s"float $id = mad($id, ${quote(a)}, ${quote(b)})")
        } else {
          cacheAndEmit(sym, s"$id = mad($id, ${quote(a)}, ${quote(b)})")
        }

      case WriteOutput(NTCell(cell, off, begin, end, gram), acc) =>
        val base = s"${quote(cell)}[("
        val ending = s"* CHART_SIZE + ${quote(off)} + TRIANGULAR_INDEX(${quote(begin)}, ${quote(end)}))*NUM_GRAMMARS + ${quote(gram)}]"
        for(id <- acc.declared) {
          cacheAndEmit(sym, s"$base$id $ending = ${acc.prefix}$id")
        }
      case _ => super.emitNode(sym, rhs)
    }

  }
}
