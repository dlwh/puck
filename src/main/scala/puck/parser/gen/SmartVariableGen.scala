package puck.parser.gen

import puck.parser.RuleStructure
import java.util._
import scala.Double
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

object SmartVariableGen {
  private val MAX_BADNESS: Int = 60
  private val MIN_BADNESS = 40
}

import SimpleGenRuleMultiply.NUM_SM

class SmartVariableGen[C, L](structure: RuleStructure[C, L], directWrite: Boolean, logSpace: Boolean) extends SimpleGenRuleMultiply[C, L](structure, directWrite, logSpace) {

  def segmentUnaries(indexedUnaryRules: List[IndexedUnaryRule[C, L]]): Array[List[IndexedUnaryRule[C, L]]] = {
    val segmentation: Array[List[IndexedUnaryRule[C, L]]] = Array(indexedUnaryRules)
    var min = Double.PositiveInfinity
    var max = Double.NegativeInfinity
    for (segment <- segmentation) {
      min = Math.min(segment.size, min)
      max = Math.max(segment.size, max)
    }
    System.out.println("min unary segment size: " + min)
    System.out.println("max unary segment size: " + max)
    segmentation
  }

  def segmentBinaries(indexedBinaryRules: List[IndexedBinaryRule[C, L]]): Array[Array[List[IndexedBinaryRule[C, L]]]] = {
    val segmentation: Array[Array[List[IndexedBinaryRule[C, L]]]] = naiveSegmentBinaries(indexedBinaryRules).map(_.map(_.asJava))
    var min = Double.PositiveInfinity
    var max = Double.NegativeInfinity
    for (segment <- segmentation) {
      for (sub <- segment) {
        min = Math.min(sub.size, min)
        max = Math.max(sub.size, max)
      }
    }
    System.out.println("min binary segment size: " + min)
    System.out.println("max binary segment size: " + max)
    segmentation
  }

  private def badness(rules: IndexedSeq[IndexedBinaryRule[C, L]], parents: Set[Integer], left: Set[Integer], right: Set[Integer]): Int = {
    left.size + right.size
  }

  private def naiveSegmentBinaries(indexedBinaryRules: List[IndexedBinaryRule[C, L]]) = {
    val segmentation: List[Array[ArrayBuffer[IndexedBinaryRule[C, L]]]] = new ArrayList[Array[ArrayBuffer[IndexedBinaryRule[C, L]]]]

    val allRules: Deque[IndexedBinaryRule[C, L]] = new ArrayDeque[IndexedBinaryRule[C, L]]
    val sortedRules: List[IndexedBinaryRule[C, L]] = new ArrayList[IndexedBinaryRule[C, L]](indexedBinaryRules)
    Collections.sort(sortedRules, new Comparator[IndexedBinaryRule[C, L]] {
      def compare(o1: IndexedBinaryRule[C, L], o2: IndexedBinaryRule[C, L]): Int = {
        val parent: Int = Integer.compare(o1.rule.parent.gpu, o2.rule.parent.gpu)
        if (parent != 0) return parent
        val lhs: Int = Integer.compare(o1.rule.left.gpu, o2.rule.left.gpu)
        if (lhs != 0) return lhs
        val rhs: Int = Integer.compare(o1.rule.right.gpu, o2.rule.right.gpu)
        rhs
      }
    })

    allRules.addAll(sortedRules)
    while (!allRules.isEmpty) {
      val segment: Array[ArrayBuffer[IndexedBinaryRule[C, L]]] = Array.fill(NUM_SM)(new ArrayBuffer[IndexedBinaryRule[C, L]]())
      segmentation.add(segment)
      var usedCoarseParents = collection.Set[Integer]()
      val usedParents: Set[Integer] = new HashSet[Integer]

      val skippedRules = new ArrayBuffer[IndexedBinaryRule[C, L]]
      for(sub <- 0 until NUM_SM) {
        val subseg = segment(sub)
        val parents: Set[Integer] = new HashSet[Integer]
        val lefts: Set[Integer] = new HashSet[Integer]
        val rights: Set[Integer] = new HashSet[Integer]
        while (!allRules.isEmpty && badness(subseg, parents, lefts, rights) < SmartVariableGen.MAX_BADNESS) {
          val rule: IndexedBinaryRule[C, L] = allRules.pop

          if ( (usedCoarseParents(rule.parent.coarse) || badness(subseg, parents, lefts, rights) < SmartVariableGen.MIN_BADNESS)) {// && !usedParents.contains(rule.parent.gpu)) {
            usedCoarseParents += rule.parent.coarse
            parents.add(rule.parent.gpu)
            lefts.add(rule.rule.left.gpu)
            rights.add(rule.rule.right.gpu)
            subseg += rule
          } else {
            skippedRules += rule
          }
        }
        usedParents.addAll(parents)
      }
      for (r <- skippedRules.reverse) allRules.push(r)
    }
    segmentation.asScala.toArray
  }


}


