package puck.parser.gen

import epic.trees.{UnaryRule, BinaryRule}
import epic.AwesomeScalaBitSet
import puck.parser.{RuleKernel, CLUnaryRuleUpdater, CLBinaryRuleUpdater, SymId}
import com.nativelibs4java.opencl.{CLKernel, CLContext}

import scala.collection.JavaConverters._
import java.util
import scala.collection.immutable.BitSet

/**
 * TODO
 *
 * @author dlwh
 **/
abstract class JavaFriendlyGenRuleMultiply[C, L] extends GenRuleMultiply[C, L] {
  def javaBinaryRuleApplication(rules: util.List[IndexedBinaryRule[C, L]], name: String, context: CLContext):CLBinaryRuleUpdater
  def javaUnaryRuleApplication(rules: util.List[IndexedUnaryRule[C, L]], name: String, context: CLContext):CLUnaryRuleUpdater

  def binaryRuleApplication(rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLBinaryRuleUpdater = {
    javaBinaryRuleApplication(rules.map((IndexedBinaryRule[C, L] _).tupled).asJava, name, cl)
  }


  def unaryRuleApplication(rules: IndexedSeq[(UnaryRule[SymId[C, L]], Int)], name: String)(implicit cl: CLContext): CLUnaryRuleUpdater = {
    javaUnaryRuleApplication(rules.map((IndexedUnaryRule[C, L] _).tupled).asJava, name, cl)
  }

  def compileKernels[T <: HasParent[C, L]](context: CLContext,
                     partitions: java.util.List[java.util.List[T]],
                     texts: java.util.List[String]):java.util.List[RuleKernel] = {
    require(partitions.size == texts.size)
    val programs = texts.asScala.map(context.createProgram(_))
    programs.foreach(_.setFastRelaxedMath())
    if(context.getDevices.head.toString.toLowerCase.contains("nvidia") && !context.getDevices.head.toString.toLowerCase.contains("apple") ) {
      programs.foreach(_.addBuildOption("-cl-nv-verbose"))
      //programs.foreach(_.addBuildOption("-cl-nv-arch"))
      //programs.foreach(_.addBuildOption("sm_30"))
    }

    (programs zip partitions.asScala).map{ case (prog, part) =>
      val coarseParents = BitSet.empty ++ getParents(part).asScala.map(_.coarse)
      RuleKernel(prog.createKernels(), coarseParents.toJavaBitSet)
    }.asJava
  }

  def getParents(partition: util.List[_ <: HasParent[C, L]] ): util.Set[SymId[C, L]] = {
    partition.asScala.map(_.parent).toSet.asJava
  }

  def supportsExtendedAtomics(context: CLContext) = {
    val ok = context.getDevices.forall(_.getExtensions.contains("cl_khr_global_int32_extended_atomics"))
    // apple can go fuck itself
    ok && !context.toString.contains("Apple")
//    ok
  }

  def flatten(arr: Array[Array[util.List[IndexedBinaryRule[C, L]]]]) = arr.map(_.map(_.asScala).toIndexedSeq.flatten.asJava).toIndexedSeq.asJava
  def flattenU(arr: Array[Array[util.List[IndexedUnaryRule[C, L]]]]) = arr.map(_.map(_.asScala).toIndexedSeq.flatten.asJava).toIndexedSeq.asJava
}

trait HasParent[C, L] {
  def parent: SymId[C, L]
}

case class IndexedBinaryRule[C, L](rule: BinaryRule[SymId[C, L]], ruleId: Int) extends java.lang.Comparable[IndexedBinaryRule[C, L]] with HasParent[C,L] {
  def parent = rule.parent
  def compareTo(o2: IndexedBinaryRule[C, L]):Int = {
    val lhs = Integer.compare(rule.left.gpu, o2.rule.left.gpu)
    if(lhs != 0) return lhs
    val rhs = Integer.compare(rule.right.gpu, o2.rule.right.gpu)
    if(rhs != 0) return rhs

    val parent = Integer.compare(rule.parent.gpu, o2.rule.parent.gpu)

    parent
  }
}

case class IndexedUnaryRule[C, L](rule: UnaryRule[SymId[C, L]], ruleId: Int) extends java.lang.Comparable[IndexedUnaryRule[C, L]] with HasParent[C, L] {
  def parent = rule.parent
  def compareTo(o2: IndexedUnaryRule[C, L]):Int = {
    val lhs = Integer.compare(rule.child.gpu, o2.rule.child.gpu)
    if(lhs != 0) return lhs

    val parent = Integer.compare(rule.parent.gpu, o2.rule.parent.gpu)
    parent
  }

}
