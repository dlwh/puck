package puck.parser

import epic.trees.{UnaryRule, BinaryRule}
import breeze.linalg.{SparseVector, Counter}
import breeze.util.{Encoder, Index}
import puck.cluster.{BalancedKMeans, KMeans}
import java.util
import breeze.stats.distributions.{ThreadLocalRandomGenerator, RandBasis}
import org.apache.commons.math3.random.MersenneTwister

/**
 * TODO
 *
 * @author dlwh
 **/
trait GrammarPartitioner[C, L] extends Serializable {
  def partition(rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)]): IndexedSeq[IndexedSeq[(BinaryRule[SymId[C, L]], Int)]]

  def partitionUnaries(rules: IndexedSeq[(UnaryRule[SymId[C, L]], Int)]): IndexedSeq[IndexedSeq[(UnaryRule[SymId[C, L]], Int)]] = {
    val partitions = partition(rules.map { case (rule, id) => BinaryRule(rule.parent, rule.child, rule.child) -> id})

    val ided = rules.map(_.swap).toMap

    for( part <- partitions) yield for ( (r, id) <- part) yield ided(id) -> id
  }
}

class KMeansGrammarPartitioner[C, L](k: Int) extends GrammarPartitioner[C, L] {
  def partition(rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)]): IndexedSeq[IndexedSeq[(BinaryRule[SymId[C, L]], Int)]] = {
    val syms = Index(rules flatMap {case (r, id) => featuresFor(r) })
    val asVectors = rules map { case (r, id) => Encoder.fromIndex(syms).encodeSparse(Counter.count(featuresFor(r):_*).mapValues(_.toDouble))}
    val ident = new util.IdentityHashMap[SparseVector[Double], (BinaryRule[SymId[C, L]], Int)]()
    for( (pair, vector) <- rules zip asVectors) {
      ident.put(vector, pair)
    }
    implicit val basis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0)))
    val clusters = new KMeans[SparseVector[Double]](k).cluster(asVectors)
    clusters.map(_ map (ident.get(_)))
  }

  def featuresFor(r: BinaryRule[SymId[C, L]]) = Seq(Left(r.left.system), Right(r.right.system), r.parent.system)
}