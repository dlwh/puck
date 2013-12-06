package puck.parser

trait RuleSemiring extends Serializable {
  def zero: Float
  def one: Float
  def times(left: String, right: String):String
  def add(left: String, right: String):String

  def plusIsIdempotent: Boolean

  def fromLogSpace(float: Float): Float
  def mad(id: String, arg1: String, arg2: String):String

}


object ViterbiRuleSemiring extends RuleSemiring {
  def zero: Float = Float.NegativeInfinity

  def one: Float = 0.0f

  def times(left: String, right: String): String = s"($left + $right)"
  def add(left: String, right: String): String = s"max($left, $right)"


  def fromLogSpace(float: Float): Float = float

  def plusIsIdempotent: Boolean = true

  def mad(id: String, arg1: String, arg2: String): String = {
    s"max($id, $arg1 + $arg2);"
  }


}