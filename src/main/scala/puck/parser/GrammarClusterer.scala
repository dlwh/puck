package puck.parser

import epic.trees.{UnaryRule, BinaryRule}
import scala.collection.immutable

/**
 * TODO
 *
 * @author dlwh
 **/
trait GrammarClusterer[C, L] extends Serializable {
  def partition(rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)]): IndexedSeq[immutable.IndexedSeq[(BinaryRule[SymId[C, L]], Int)]]

  def partitionUnaries(rules: IndexedSeq[(UnaryRule[SymId[C, L]], Int)]): IndexedSeq[immutable.IndexedSeq[(UnaryRule[SymId[C, L]], Int)]] = {
    val partitions = partition(rules.map { case (rule, id) => BinaryRule(rule.parent, rule.child, rule.child) -> id})

    val ided = rules.map(_.swap).toMap

    for( part <- partitions) yield for ( (r, id) <- part) yield ided(id) -> id
  }
}

