package puck.parser.gen

import epic.trees.{UnaryRule, BinaryRule}
import puck.parser.{CLUnaryRuleUpdater, CLBinaryRuleUpdater, SymId}
import com.nativelibs4java.opencl.{CLKernel, CLContext}

import scala.collection.JavaConverters._
import java.util

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

  def compileKernels(context: CLContext, texts: java.util.List[String]):java.util.List[CLKernel] = {
    val programs = texts.asScala.map(context.createProgram(_))
    programs.foreach(_.setFastRelaxedMath())
    if(context.getDevices.head.toString.toLowerCase.contains("nvidia") && !context.getDevices.head.toString.toLowerCase.contains("apple") ) {
      programs.foreach(_.addBuildOption("-cl-nv-verbose"))
      //programs.foreach(_.addBuildOption("-cl-nv-arch"))
      //programs.foreach(_.addBuildOption("sm_30"))
    }

//    programs.par.flatMap(_.createKernels()).seq.asJava
    programs.flatMap(_.createKernels()).asJava
  }

  def getParents(partition: util.List[IndexedBinaryRule[C, L]] ): util.Set[SymId[C, L]] = {
    partition.asScala.map(_.rule.parent).toSet.asJava
  }
}

case class IndexedBinaryRule[C, L](rule: BinaryRule[SymId[C, L]], ruleId: Int) extends java.lang.Comparable[IndexedBinaryRule[C, L]] {
  def compareTo(o2: IndexedBinaryRule[C, L]):Int = {
    val lhs = Integer.compare(rule.left.gpu, o2.rule.left.gpu)
    if(lhs != 0) return lhs
    val rhs = Integer.compare(rule.right.gpu, o2.rule.right.gpu)
    if(rhs != 0) return rhs

    val parent = Integer.compare(rule.parent.gpu, o2.rule.parent.gpu)

    parent
  }
}

case class IndexedUnaryRule[C, L](rule: UnaryRule[SymId[C, L]], ruleId: Int) extends java.lang.Comparable[IndexedUnaryRule[C, L]] {
  def compareTo(o2: IndexedUnaryRule[C, L]):Int = {
    val lhs = Integer.compare(rule.child.gpu, o2.rule.child.gpu)
    if(lhs != 0) return lhs

    val parent = Integer.compare(rule.parent.gpu, o2.rule.parent.gpu)
    parent
  }

}
