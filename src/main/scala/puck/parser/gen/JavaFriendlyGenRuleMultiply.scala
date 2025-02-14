package puck.parser.gen

import puck.parser._
import com.nativelibs4java.opencl.CLContext

import scala.collection.JavaConverters._
import epic.AwesomeBitSet
import java.util
import epic.trees.BinaryRule
import epic.trees.UnaryRule
import puck.parser.SymId
import breeze.linalg.DenseVector
import puck.util.BitHacks
import java.io.{FileOutputStream, File}

/**
 * TODO
 *
 * @author dlwh
 **/
abstract class JavaFriendlyGenRuleMultiply[C, L](structure: RuleStructure[C, L], writeDirectToChart: Boolean) extends GenRuleMultiply[C, L] {
  val WARP_SIZE = 32;
  val NUM_WARPS = 48;
  val NUM_SM = 8;
  def javaBinaryRuleApplication(rules: util.List[IndexedBinaryRule[C, L]], name: String, context: CLContext, lt: LoopType):CLBinaryRuleUpdater
  def javaUnaryRuleApplication(rules: util.List[IndexedUnaryRule[C, L]], name: String, context: CLContext):CLUnaryRuleUpdater

  def binaryRuleApplication(rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)], name: String, loopType: LoopType)(implicit cl: CLContext): CLBinaryRuleUpdater = {
    javaBinaryRuleApplication(rules.map((IndexedBinaryRule[C, L] _).tupled).asJava, name, cl, loopType)
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
      programs.foreach(_.addBuildOption("-DNVIDIA"))
//      programs.foreach(_.addBuildOption("-cl-nv-opt-level=0"))
//      programs.foreach(_.addBuildOption("-cl-opt-disable"))
//      programs.foreach(_.addBuildOption("-cl-nv-arch"))
      //programs.foreach(_.addBuildOption("sm_30"))
//      programs.foreach(_.addBuildOption("sm_35"))

    }

    val result = (programs zip partitions.asScala).map{ case (prog, part) =>
      val mask = new Array[Int](puck.roundUpToMultipleOf(structure.numCoarseSyms, 32) / 32)
      for(p <- getParents(part).asScala.map(_.coarse)) {
        mask(p/32) |= 1<<(p%32)
      }
      val globalSize = Array(WARP_SIZE * NUM_WARPS, NUM_SM, 1);
      val wgSize = Array(WARP_SIZE * 3, 1, 1);
      RuleKernel(prog.createKernels(), part.asScala.toIndexedSeq, globalSize, wgSize, new DenseVector(mask))
    }.asJava




    for( (r, t) <- result.asScala.zip(texts.asScala)) {
      val srcOut = new File(s"genSources/${structure.numNonTerms}/${r.kernels.head.getFunctionName}.cl")
      srcOut.getParentFile.mkdirs()
      val oout = new FileOutputStream(srcOut)
      oout.write(t.getBytes("UTF-8"))
      oout.close()
      for ((plat, ptx) <-  r.kernels.head.getProgram.getBinaries.asScala) {
        val out = new File(s"compiled/${r.kernels.head.getFunctionName}_${plat.toString.filter(_.isLetterOrDigit)}.ptx")
        out.getParentFile.mkdirs()
        val oout = new FileOutputStream(out)
        oout.write(ptx)
        oout.close()

      }
      val names = r.kernels.map(_.getFunctionName)
      val parents = BitHacks.asBitSet(r.parents).iterator.map(structure.refinements.labels.coarseIndex.get(_:Int)).toSet
      val ruleCounts = r.rules.length
      //println(names,parents,ruleCounts)
    }

    result
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
