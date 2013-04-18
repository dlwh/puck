package puck.parser.gen

import scala.virtualization.lms.common.{VariablesExp, BaseExp, Variables, Base}
import scala.reflect.SourceContext
import scala.virtualization.lms.internal.Effects

/**
 *
 *
 * @author dlwh
 */
trait AccumulatorOps extends SemiringOps  { self : Base with Variables =>
  trait AccumulatorBase {
    def update(sym: Int, score: Rep[Real]):Rep[Unit]
    def apply(sym: Int):Var[Real]
    def mad(sym: Int, score1: Rep[Real], score2: Rep[Real]): Rep[Unit]
  }
  type Accumulator <: AccumulatorBase

  implicit class AccOps(acc: Accumulator) {
  }

  def accumulator(ids: Set[Int])(implicit pos: SourceContext):Accumulator

}


trait AccumulatorOpsExp extends AccumulatorOps with VariablesExp { self: BaseExp with Variables with Effects =>
  case class Accumulator(ids: Set[Int]) extends AccumulatorBase {
    val vars = ids.toArray.map{ i => i -> var_new(zero)}.toMap
    def apply(sym: Int) = vars(sym)
    def update(sym: Int, score: Rep[Real]) = {self.__assign(apply(sym), score) }
    def mad(sym: Int, score1: Rep[Real], score2: Rep[Real]): Rep[Unit] = update(sym, self.mad(apply(sym), score1, score2))
  }

  def accumulator(ids: Set[Int])(implicit pos: SourceContext) = new Accumulator(ids)
}
