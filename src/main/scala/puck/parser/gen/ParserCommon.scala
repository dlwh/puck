package puck.parser.gen

import virtualization.lms.common._
import trochee.kernels.{KernelOpsExp, KernelOps}
import spire.algebra.Rig
import reflect.SourceContext
import virtualization.lms.common.RangeOpsExp
import virtualization.lms.common.RangeOps
import virtualization.lms.common.Base
import virtualization.lms.common.IfThenElseExp
import trochee.util.CStructExp
import trochee.basic._
import scala.virtualization.lms.util.OverloadHack
import scala.collection.mutable.ArrayBuffer
import scala.virtualization.lms.internal.Effects
import epic.trees.{BinaryRule, Rule}
import puck.newparser.generator.RuleStructure

/**
 * 
 * @author dlwh
 */
trait ParserCommon[L] extends ExtraBase with AccumulatorOps with SpireOps with OverloadHack with BooleanOps { self: Base with KernelOps with RangeOps =>

  def numGrammars: Int


  type ParseChart
  implicit def manifestParseChart: Manifest[ParseChart]
  def infix_update(chart: Rep[ParseChart], offset: Rep[Int], begin: Rep[Int], end: Rep[Int], gram: Rep[Int], acc: Accumulator): Rep[Unit]
  def infix_apply(chart: Rep[ParseChart], offset: Rep[Int], begin: Rep[Int], end: Rep[Int], gram: Rep[Int]):ParseCell

  implicit class RichParseChart(chart: Rep[ParseChart]) {
    def apply(offset: Rep[Int], begin: Rep[Int], end: Rep[Int], gram: Rep[Int]) = infix_apply(chart, offset, begin, end, gram)
    def update(offset: Rep[Int], begin: Rep[Int], end: Rep[Int], gram: Rep[Int], acc: Accumulator) = infix_update(chart, offset, begin, end, gram, acc)
  }

  type ParseCell
  def infix_apply(cell: ParseCell, sym: Rep[Int])(implicit pos: SourceContext):Rep[Real]
  def infix_update(cell: ParseCell, sym: Rep[Int], v: Rep[Real])(implicit pos: SourceContext):Rep[Unit]
  def infix_+=(cell: ParseCell, acc: Accumulator)(implicit pos: SourceContext):Rep[Unit]

  implicit class RichParseCell(cell: ParseCell) {
    def apply(sym: Rep[Int])(implicit pos: SourceContext) = infix_apply(cell, sym)(pos)
    def update(sym: Rep[Int], v: Rep[Real])(implicit pos: SourceContext) = infix_update(cell, sym, v)(pos)
    def +=(acc: Accumulator)(implicit pos: SourceContext) = infix_+=(cell, acc)(pos)
  }

  /*
  typedef struct {
    float syms[NUM_SYMS][NUM_GRAMMARS];
  } parse_cell;
 */

  type TermChart
  implicit def manifestTermChart: Manifest[TermChart]
  def infix_update(chart: Rep[TermChart], offset: Rep[Int], pos: Rep[Int], gram: Rep[Int], acc: Accumulator): Rep[Unit]
  def infix_apply(chart: Rep[TermChart], offset: Rep[Int], pos: Rep[Int], gram: Rep[Int]):ParseCell

  implicit class RichTermChart(chart: Rep[TermChart]) {
    def apply(offset: Rep[Int], begin: Rep[Int],  gram: Rep[Int]) = infix_apply(chart, offset, begin, gram)
    def update(offset: Rep[Int], begin: Rep[Int], gram: Rep[Int], acc: Accumulator) = infix_update(chart, offset, begin, gram, acc)
  }
  

  type RuleCell
  implicit def manifestRuleCell: Manifest[RuleCell]
  def infix_rules(cell: Rep[RuleCell], r: Rep[Int], g: Rep[Int]):Rep[Real]

  def grammar: RuleStructure[_, L]
  def accumulatorForParents(rules: IndexedSeq[(Rule[Int], Int)]) = accumulator(rules.map(_._1.parent).toSet)
  def accumulatorForChildren(rules: IndexedSeq[(Rule[Int], Int)]) = accumulator(rules.flatMap(_._1.children).toSet)
  def accumulatorForRightChildren(rules: IndexedSeq[(BinaryRule[Int], Int)]) = accumulator(rules.map(_._1.right).toSet)
  def accumulatorForLeftChildren(rules: IndexedSeq[(BinaryRule[Int], Int)]) = accumulator(rules.map(_._1.left).toSet)
  def accumulatorForRules(rules: IndexedSeq[(BinaryRule[Int], Int)]) = accumulator(rules.map(_._2).toSet)

  def numSyms: Int = grammar.numSyms


  def expprintf(string: String, args: Rep[_]*):Rep[Unit]
}

trait ParserCommonExp[L] extends ParserCommon[L] with BaseFatExp with CStructExp with KernelOpsExp with Effects with AccumulatorOpsExp with BooleanOpsExp { self: Base with RangeOpsExp with IfThenElseExp =>


  sealed trait ParseCell { def chart: Rep[_]}
  final case class NTCell(chart: Rep[ParseChart], offset: Rep[Int], begin: Rep[Int], end: Rep[Int], gram: Rep[Int]) extends ParseCell
  final case class TCell(chart: Rep[TermChart], offset: Rep[Int], pos: Rep[Int], gram: Rep[Int]) extends ParseCell

  def manifestParseChart: Manifest[ParseChart] = implicitly
  def manifestRuleCell: Manifest[RuleCell] = implicitly
  def manifestTermChart: Manifest[TermChart] = implicitly


  trait RuleCell
  trait ParseChart
  trait TermChart
  def expprintf(string: String, args: Rep[_]*):Rep[Unit] = reflectEffect(Printf(string, args))


  lazy val structy = new CStruct { val rules: Array[Array[Real]] = Array.ofDim[Real](grammar.numRules, numGrammars)}

  override def register() {
    define("PARSE_CELL", "float*")
    define("NUM_GRAMMARS", numGrammars)
    define("NUM_RULES", grammar.numRules)
    define("TRIANGULAR_INDEX(begin,end)","((end) *((end)+1)/2) + (begin)")
    struct("rule_cell", structy)
    super[KernelOpsExp].register()
  }


    
  def infix_rules(cell: Rep[RuleCell], r: Rep[Int], g: Rep[Int]):Rep[Real] = RuleDeref(cell, r, g)

  case class RuleDeref(cell: Rep[RuleCell], r: Rep[Int], g: Rep[Int]) extends Exp[Real]


  case class CellApply(cell: ParseCell, sym: Rep[Int])(implicit val pos: SourceContext) extends Def[Real]
  case class CellUpdate(cell: ParseCell, sym: Rep[Int], v: Exp[Real]) extends Def[Unit]

  def infix_update(chart: Rep[ParseChart], offset: Rep[Int], begin: Rep[Int], end: Rep[Int], gram: Rep[Int], acc: Accumulator): Rep[Unit] = {
    for((id, vr) <- acc.vars) {
      reflectWrite(chart)(CellUpdate(NTCell(chart, offset, begin, end, gram), id, ReadVar(vr)))
    }
    //reflectWrite(chart)(WriteOutput(NTCell(chart, offset, begin, end, gram), acc))
  }

  def infix_apply(chart: Rep[ParseChart], offset: Rep[Int], begin: Rep[Int], end: Rep[Int], gram: Rep[Int]): ParseCell = NTCell(chart, offset, begin, end, gram)

  def infix_apply(cell: ParseCell, sym: Rep[Int])(implicit pos: SourceContext): Rep[Real] = CellApply(cell, sym)(pos)

  def infix_update(chart: Rep[TermChart], offset: Rep[Int], pos: Rep[Int], gram: Rep[Int], acc: Accumulator): Rep[Unit] = {
    for((id, vr) <- acc.vars) {
      reflectWrite(chart)(CellUpdate(TCell(chart, offset, pos, gram), id, ReadVar(vr)))
    }
  }

  def infix_update(chart: ParseCell, sym: Rep[Int], v: Rep[Real])(implicit pos: SourceContext): Rep[Unit] = {
    reflectWrite(chart.chart)(CellUpdate(chart, sym, v))
  }

  def infix_+=(chart: ParseCell, acc: Accumulator)(implicit pos: SourceContext): Rep[Unit] = {//reflectEffect(WriteOutput(TCell(chart, offset, pos, gram), acc), infix_andAlso(Simple(), Write(List(chart.asInstanceOf[Sym[Any]])) ))
    for((id, vr) <- acc.vars) {
      reflectWrite(chart.chart)(CellUpdate(chart, id,  rig_plus(toAtom(CellApply(chart, id)(pos)):Exp[Real], ReadVar(vr))))
    }
  }

  def infix_apply(chart: Rep[TermChart], offset: Rep[Int], pos: Rep[Int], gram: Rep[Int]) = TCell(chart, offset, pos, gram)


}
