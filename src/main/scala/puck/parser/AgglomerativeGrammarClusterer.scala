package puck.parser

import scala.collection.{immutable, mutable}
import collection.immutable.BitSet
import epic.trees.BinaryRule

import GrammarClusterer._
import com.typesafe.scalalogging.log4j.Logging

class AgglomerativeGrammarClusterer[C, L](grouper: Int=>Int = identity, numRestarts: Int = 100, maxPartitionLabelSize: Int = 128) extends GrammarClusterer[C, L] with Logging {


  def partition(rules: IndexedSeq[(BinaryRule[SymId[C, L]], Int)],
                targetLabel: TargetLabel = Parent): IndexedSeq[immutable.IndexedSeq[(BinaryRule[SymId[C, L]], Int)]] = {


    val clusters_x: Map[Int, IndexedSeq[(BinaryRule[SymId[C, L]], Int)]] = rules.groupBy(r => grouper(targetLabel.target(r._1)))

    val initialClusters = clusters_x.map { case (p:Int, r: IndexedSeq[(BinaryRule[SymId[C, L]], Int)]) =>
      val (g1, g2) = r.map(rr => targetLabel.clusterPieces(rr._1)).unzip
      p -> Partition(BitSet(p), g1.reduce( _ ++ _), g2.reduce(_ ++ _))
    }

    val clusters = (0 until numRestarts).par.aggregate(restart(initialClusters, maxPartitionLabelSize, 1.0))({ (c1, seed) =>
        val r = new java.util.Random(seed)
        val c2 = restart(initialClusters, maxPartitionLabelSize,.3 +.7 * r.nextDouble)
        if (c1.values.map(_.badness).sum < c2.values.map(_.badness).sum) c1 else c2
    }, {
      (c1, c2) => if (c1.values.map(_.badness).sum < c2.values.map(_.badness).sum) c1 else c2
    })

    println("Best badness: " + targetLabel  + " " + clusters.values.iterator.map(_.badness).sum)

    var p = 0
    for( Partition(targets, g1, g2, _) <- clusters.values.iterator) {
      println("Partition " + p)
      println("G1: " + g1.size + " " + g1)
      println("G2: " + g2.size + " "  + g2)
      println("targets: " + targets)
      p += 1
    }

    assert(clusters.values.iterator.flatMap(_.targets).toSet.size == clusters_x.keySet.size)
    clusters.values.iterator.map(p => p.targets.flatMap(clusters_x).toIndexedSeq).toIndexedSeq
  }


  private def restart(initialClusters: Map[Int, Partition], maxPartitionLabelSize: Int, random: =>Double) = {

    var clusters = initialClusters.map { case (k,v) => BitSet(k) -> v}

    def remove(p: Partition, t: Int) = {
      (for(t2 <- p.targets if t != t2) yield initialClusters(t2)).reduceLeft(_.merge(_, maxPartitionLabelSize).get)
    }

    sealed trait Action { def priority: Double}
    case class Merge(p1: Partition, p2: Partition, merged: Partition) extends Action {
      val priority = (p1.badness + p2.badness - merged.badness)*random
    }
    case class SplitMerge(p1: Partition, p2: Partition, t: Int) extends Action {
      val newP1 = remove(p1, t)
      val newP2 = p2.merge(initialClusters(t), maxPartitionLabelSize)
      val ok = newP2.nonEmpty
      val priority = if(!ok) Double.NegativeInfinity else (p1.badness + p2.badness - newP1.badness - newP2.get.badness)*random
    }


    implicit val order = Ordering[Double].on[Action](_.priority)

    val queue = new mutable.PriorityQueue[Action]
    queue ++= {
      val seqqed: immutable.IndexedSeq[Partition] = clusters.values.toIndexedSeq
      for ( IndexedSeq(p1:Partition, p2: Partition) <- seqqed.combinations(2); merged <- p1.merge(p2, maxPartitionLabelSize)) yield Merge(p1, p2, merged)
    }

    while(queue.nonEmpty) {
      queue.dequeue() match {
        case sm@Merge(l, r, merger) =>
          if(clusters.contains(l.targets) && clusters.contains(r.targets)) {
              clusters -= l.targets
              clusters -= r.targets
              queue ++= {for(p2 <- clusters.values.iterator; merged <- merger.merge(p2, maxPartitionLabelSize)) yield Merge(merger, p2, merged)}
              //                queue ++= {for(p2 <- clusters.values.iterator; rm  <- merger.targets) yield SplitMerge(merger, p2, rm)}
              clusters += (merger.targets -> merger)
          }
        case sm@SplitMerge(l, r, _) =>
          /*
          if(clusters.contains(l.targets) && clusters.contains(r.targets)) {
            import sm._
              clusters -= l.targets
              clusters -= r.targets
              queue ++= {for(p2 <- clusters.values.iterator; merged <- .merge(p2, maxPartitionLabelSize)) yield Merge(newP1, p2, newP1 merge p2)}
              queue ++= {for(p2 <- clusters.values.iterator; merged <- merger.merge(p2, maxPartitionLabelSize)) yield Merge(newP2, p2, newP2 merge p2)}
              //                queue ++= {for(p2 <- clusters.values.iterator; rm  <- newP1.targets if newP1.targets.size > 1) yield SplitMerge(newP1, p2, rm)}
              //                queue ++= {for(p2 <- clusters.values.iterator; rm  <- newP2.targets if newP2.targets.size > 1) yield SplitMerge(newP2, p2, rm)}
              clusters += (newP1.targets -> newP1)
              clusters += (newP2.targets -> newP2)

            }
          }
          */
      }
    }

    clusters
  }

  case class Partition(targets: BitSet, group1: BitSet, group2: BitSet, isPure: Boolean = true) {
    def merge(p: Partition, maxSize: Int) = {
      Some(Partition(targets | p.targets, group1 | p.group1, group2 | p.group2, false))
        .filter(np => !np.isTooBig(maxSize)  || np.badness <= math.max(this.badness, p.badness))
    }

    def tSize = targets.size

    def badness = group1.size + group2.size


    def isTooBig(maxSize: Int) = !isPure && (group1.size + group2.size + targets.size) >maxSize
  }

}