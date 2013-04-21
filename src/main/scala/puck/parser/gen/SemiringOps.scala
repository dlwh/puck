package puck.parser.gen

import scala.virtualization.lms.common.{BaseExp, BaseFatExp, Variables, Base}
import scala.reflect.SourceContext
import trochee.basic.{SpireOpsExp, SpireOps}
import scala.reflect.runtime.universe._

/**
 *
 * @author dlwh
 */
trait SemiringOps extends SpireOps { this: Base with Variables =>
  type Real
  def _zero: Real
  def _one: Real
  def zero: Rep[Real]
  def one: Rep[Real]
  implicit def manifestReal: Manifest[Real]
  implicit def ttReal: TypeTag[Real]

//  implicit def rigRepReal : Rig[Rep[Real]]

  implicit def repRealToRigNumeric(n: Rep[Real])(implicit pos: SourceContext) = new RigOpsCls(n)

  def mad(a: Rep[Real], b: Rep[Real], c: Rep[Real])(implicit pos: SourceContext):Rep[Real]

  class RigOpsCls(lhs: Rep[Real])(implicit pos: SourceContext) {
    def +[A](rhs: A)(implicit c: A => Real, pos: SourceContext) = rig_plus(lhs,unit(c(rhs)))(pos)
    def +(rhs: Rep[Real])(implicit pos: SourceContext) = rig_plus(lhs,rhs)(pos)
    def *(rhs: Rep[Real])(implicit pos: SourceContext) = rig_times(lhs,rhs)(pos)
    def toLogSpace:Rep[Float] = rig_logSpace(lhs)(pos)
  }

  def rig_plus(lhs: Rep[Real], rhs: Rep[Real])(implicit pos: SourceContext): Rep[Real]
  def rig_times(lhs: Rep[Real], rhs: Rep[Real])(implicit pos: SourceContext): Rep[Real]
  def rig_logSpace(a: Rep[Real])(implicit pos: SourceContext):Rep[Float]
  def fromLogSpace(a: Float):Real
}

trait FloatOpsExp { this: BaseExp =>
  case class Mad(a: Rep[Float], b: Rep[Float], c: Rep[Float])(implicit pos: SourceContext) extends Def[Float]
  case class Log(a: Rep[Float])(implicit pos: SourceContext) extends Def[Float]
  case class LogAdd(a: Rep[Float], b: Rep[Float])(implicit pos: SourceContext) extends Def[Float]
}


trait SemringFloatOpsExp extends SemiringOps with SpireOpsExp with FloatOpsExp {  this: BaseExp with BaseFatExp with Variables =>
  type Real = Float
  def manifestReal = manifest[Real]
  def ttReal = implicitly
  def zero: Rep[Real] = unit(0.0f)
  def one : Rep[Real] = unit(1.0f)

  def _zero = 0.0f
  def _one = 1.0f

  def mad(a: Rep[Real], b: Rep[Real], c: Rep[Real])(implicit pos: SourceContext):Rep[Real] = Mad(a, b, c)(pos)


  def rig_plus(lhs: Rep[Real], rhs: Rep[Real])(implicit pos: SourceContext): Rep[Real] = {
    numeric_plus(lhs, rhs)(implicitly, implicitly, pos)
  }
  def rig_times(lhs: Rep[Real], rhs: Rep[Real])(implicit pos: SourceContext): Rep[Real] = {
    numeric_times(lhs, rhs)(implicitly, implicitly, pos)
  }
  def rig_logSpace(a: Rep[Real])(implicit pos: SourceContext):Rep[Float] = {
    Log(a)(pos)
  }

  def fromLogSpace(a: Float):Real = math.exp(a).toFloat
}

trait LogSpaceFloatOpsExp extends SemiringOps with SpireOpsExp with FloatOpsExp {  this: BaseExp with BaseFatExp with Variables =>
  type Real = Float
  def manifestReal = manifest[Real]
  def ttReal = implicitly
  def zero: Rep[Real] = unit(Float.NegativeInfinity)
  def one : Rep[Real] = unit(0.0f)

  def _zero = Float.NegativeInfinity
  def _one = 0.0f

  def mad(a: Rep[Real], b: Rep[Real], c: Rep[Real])(implicit pos: SourceContext):Rep[Real] = {
    rig_plus(a, rig_times(b,c))(pos)
  }


  def rig_plus(lhs: Rep[Real], rhs: Rep[Real])(implicit pos: SourceContext): Rep[Real] = {
    LogAdd(lhs, rhs)(pos)
  }

  def rig_times(lhs: Rep[Real], rhs: Rep[Real])(implicit pos: SourceContext): Rep[Real] = {
    numeric_plus(lhs, rhs)(implicitly, implicitly, pos)
  }

  def rig_logSpace(a: Rep[Real])(implicit pos: SourceContext):Rep[Float] = {
    a
  }

  def fromLogSpace(a: Float):Real = a

}