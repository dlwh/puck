package puck.newparser

import trochee.codegen.{OpenCLKernelGenVariables, OpenCLKernelCodegen}
import trochee.basic.{SpireOpsExp, GenSpireOps}
import scala.virtualization.lms.internal.{FatExpressions, Effects, Expressions}
import puck.parser.gen.{FloatOpsExp, ParserCommonExp}
import scala.virtualization.lms.common.IfThenElseExp

/**
 * TODO
 *
 * @author dlwh
 **/
trait OpenCLParserGen[L] extends OpenCLKernelCodegen with OpenCLKernelGenVariables with GenSpireOps {
  val IR: Expressions with Effects with FatExpressions with ParserCommonExp[L] with trochee.kernels.KernelOpsExp with IfThenElseExp with SpireOpsExp with FloatOpsExp
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
        emitValDef(sym, s"mad(${quote(a)}, ${quote(b)}, ${quote(c)})")
      case LogAdd(a, b) =>
        emitValDef(sym, s"${quote(a)} == -INFINITY ? ${quote(b)} : (${quote(a)} + log(1.0f + exp(${quote(b)} - ${quote(a)})))")
      case Log(a) =>
        cacheAndEmit(sym, s"log(${quote(a)})")
      case BooleanNegate(b) => emitValDef(sym, "!" + quote(b))
      case BooleanAnd(lhs,rhs) => emitValDef(sym, quote(lhs) + " && " + quote(rhs))
      case BooleanOr(lhs,rhs) => emitValDef(sym, quote(lhs) + " || " + quote(rhs))
      case Printf(str, args) =>
        val call = s"""printf(${(quote(str) +: args.map(quote _)).mkString(", ")});"""
        cacheAndEmit(sym, call)
      case _ => super.emitNode(sym, rhs)
    }

  }
}
