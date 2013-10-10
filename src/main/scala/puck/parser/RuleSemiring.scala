package puck.parser

trait RuleSemiring extends Serializable {
  def zero: Float
  def one: Float
  def times(left: String, right: String):String
  def add(left: String, right: String):String

  def accumulator(ids: Set[Int]):Accumulator

  def timesIsIdempotent: Boolean

  def fromLogSpace(float: Float): Float

  trait Accumulator {
    def ids: Set[Int]
    def declare: String
    def mad(id: Int, arg1: String, arg2: String):String
    def output(idSink: Int=>String):String
  }
}


object ViterbiRuleSemiring extends RuleSemiring {
  def zero: Float = Float.NegativeInfinity

  def one: Float = 0.0f

  def times(left: String, right: String): String = s"($left + $right)"
  def add(left: String, right: String): String = s"max($left, $right)"


  def fromLogSpace(float: Float): Float = float

  def timesIsIdempotent: Boolean = true

  def zeroString = if(zero == Float.NegativeInfinity) "-INFINITY" else zero.toString

  def accumulator(ids: Set[Int]): ViterbiRuleSemiring.Accumulator = {
    val _ids = ids
    new Accumulator {
      def mad(id: Int, arg1: String, arg2: String): String = s"parent_$id = max(parent_$id, $arg1 + $arg2)"

      def ids: Set[Int] = _ids

      def declare: String = {for(id <- ids) yield s"parent_$id"}.mkString("float ", s" = $zeroString, ",s"= $zeroString;")

      def output(idSink: (Int) => String): String = {
        for(id <- ids) yield {
          s"${idSink(id)} = parent_$id;"
        }
      }.mkString(" ")
    }
  }
}