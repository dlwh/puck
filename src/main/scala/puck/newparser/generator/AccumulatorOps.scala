package puck.newparser.generator

import scala.virtualization.lms.common.{VariablesExp, BaseExp, Variables, Base}
import scala.reflect.SourceContext
import scala.virtualization.lms.internal.Effects
import puck.parser.gen.{SemiringOps, FloatOpsExp}

/**
 *
 *
 * @author dlwh
 */
trait AccumulatorOps extends SemiringOps { self : Base with Variables =>
  def mad(a: Rep[Real], b: Rep[Real], c: Rep[Real])(implicit pos: SourceContext):Rep[Real]
  trait AccumulatorBase {
    def update(sym: Int, score: Rep[Real]):Rep[Unit]
    def apply(sym: Int):Var[Real]
    def mad(sym: Int, score1: Rep[Real], score2: Rep[Real]): Rep[Unit]
    def foreachUsed[A](f: (Int, Rep[Real])=>Rep[A]):Rep[Unit]
  }
  type Accumulator <: AccumulatorBase

  implicit class AccOps(acc: Accumulator) {
  }

  def accumulator(ids: Set[Int])(implicit pos: SourceContext):Accumulator

}


trait AccumulatorOpsExp extends AccumulatorOps with VariablesExp { self: BaseExp with Variables with Effects =>
  case class Accumulator(ids: Set[Int])(implicit pos: SourceContext) extends AccumulatorBase {
    val vars: Map[Int, Var[Real]] = ids.toArray.map{ i => i -> var_new(zero)(implicitly, pos)}.toMap
    def apply(sym: Int) = vars(sym)
    var touched = Set.empty[Int]
    def mad(sym: Int, score1: Rep[Real], score2: Rep[Real]): Rep[Unit] = update(sym, self.mad(score1, score2, apply(sym)))
    def update(sym: Int, score: Rep[Real]) = {touched += sym; self.__assign(apply(sym), score) }
    def foreachUsed[A](f: (Int, Rep[Real])=>Rep[A]):Rep[Unit] = for(i <- touched) f(i, vars(i))
  }

  def accumulator(ids: Set[Int])(implicit pos: SourceContext) = new Accumulator(ids)(pos)
}
