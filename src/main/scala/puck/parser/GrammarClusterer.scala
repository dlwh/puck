package puck.parser

import epic.trees.BinaryRule
import scala.collection.immutable.BitSet
import puck.parser.GrammarClusterer._
import scala.collection.immutable

/**
 * TODO
 *
 * @author dlwh
 **/
trait GrammarClusterer {

  def partition(rules: IndexedSeq[(BinaryRule[Int], Int)],
                maxPartitionLabelSize: Int = 100,
                numRestarts: Int = 100,
                targetLabel: TargetLabel = Parent): IndexedSeq[immutable.IndexedSeq[(BinaryRule[Int], Int)]]

}

object GrammarClusterer {
  sealed trait TargetLabel {
    def clusterPieces(r: BinaryRule[Int]) = this match {
      case Parent => BitSet(r.left) -> BitSet(r.right)
      case LeftChild => BitSet(r.parent) -> BitSet(r.right)
      case RightChild => BitSet(r.parent) -> BitSet(r.left)
    }

    def target(r: BinaryRule[Int]) = this match {
      case Parent => r.parent
      case LeftChild => r.left
      case RightChild => r.right
    }
  }
  case object Parent extends TargetLabel
  case object LeftChild extends TargetLabel
  case object RightChild extends TargetLabel



}
