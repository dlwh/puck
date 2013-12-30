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
    if(context.getDevices.head.toString.toLowerCase.contains("nvidia") && !context.getDevices.head.toString.toLowerCase.contains("apple") )
      programs.foreach(_.addBuildOption("-cl-nv-verbose"))

//    programs.par.flatMap(_.createKernels()).seq.asJava
    programs.flatMap(_.createKernels()).asJava
  }

  def getParents(partition: util.List[IndexedBinaryRule[C, L]] ): util.Set[SymId[C, L]] = {
    partition.asScala.map(_.rule.parent).toSet.asJava
  }

  def supportsExtendedAtomics(context: CLContext) = {
    val ok = context.getDevices.forall(_.getExtensions.contains("cl_khr_global_int32_extended_atomics"))
    // apple can go fuck itself
    ok && !context.toString.contains("Apple")
  }
}

case class IndexedBinaryRule[C, L](rule: BinaryRule[SymId[C, L]], ruleId: Int)
case class IndexedUnaryRule[C, L](rule: UnaryRule[SymId[C, L]], ruleId: Int)
