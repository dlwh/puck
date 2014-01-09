package puck.parser

trait RuleSemiring extends Serializable {
  def zero: Float
  def one: Float
  def times(left: String, right: String):String
  def add(left: String, right: String):String

  def plusIsIdempotent: Boolean

  def fromLogSpace(float: Float): Float

  def toLogSpace(float: Float, scale: Float):Float

  def includes: String

  def needsScaling: Boolean = false

}


object ViterbiRuleSemiring extends RuleSemiring {
  def zero: Float = Float.NegativeInfinity

  def one: Float = 0.0f

  def times(left: String, right: String): String = s"($left + $right)"
  def add(left: String, right: String): String = s"max($left, $right)"


  def fromLogSpace(float: Float): Float = float

  def plusIsIdempotent: Boolean = true
  def toLogSpace(float: Float, scale: Float):Float = float


  def includes: String =
    "inline float semiring_mad(float x, float y, float z) {\n" +
  "	return max(x, y + z);\n" +
  "}" +
      "" +
      "inline float semiring_add(float x, float y) { return max(x,y); }\n\n\n"
}

object LogSumRuleSemiring extends RuleSemiring {
  def zero: Float = -1000000.0f

  def one: Float = 0.0f

  def times(left: String, right: String): String = s"($left + $right)"
  def add(left: String, right: String): String = s"max($left, $right)"


  def fromLogSpace(float: Float): Float = float

  def plusIsIdempotent: Boolean = false

  def toLogSpace(float: Float, scale: Float):Float = float

  def includes: String = "inline float semiring_mad(float x, float _y, float z) {\n " +
    "  float y = _y + z;\n	" +
    "float tmp = x;\n" +
    "	x = min(x, y);\n" +
    "	y = max(tmp, y);\n" +
    "	return y + native_log(1.0f + native_exp(x - y));\n" +
    "}" +
    "" +
    "inline float semiring_add(float x, float y) {\n" +
    "  float tmp = x;\n" +
    "	 x = min(x, y);\n" +
    "	 y = max(tmp, y);\n" +
    "	 return y + native_log(1.0f + native_exp(x - y));\n" +
    "}" +
    "\n\n\n"

}


object RealSemiring extends RuleSemiring {
  def zero: Float = 0.0f

  def one: Float = 1.0f

  def times(left: String, right: String): String = s"($left * $right)"
  def add(left: String, right: String): String = s"($left + $right)"

  def fromLogSpace(float: Float): Float = math.exp(float).toFloat


  def toLogSpace(float: Float, scale: Float): Float = math.log(float).toFloat + scale

  def plusIsIdempotent: Boolean = false

  def includes= "inline float semiring_mad(float x, float y, float z) {\n" + "	return mad(y, z, x);\n" + "} inline float semiring_add(float x, float y) { return x + y; }\n\n\n"


  override def needsScaling: Boolean = true
}