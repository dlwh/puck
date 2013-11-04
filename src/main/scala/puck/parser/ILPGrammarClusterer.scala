package puck.parser

import epic.trees.BinaryRule
import puck.parser.GrammarClusterer.TargetLabel
import scala.collection.immutable
import breeze.optimize.linear.LinearProgram
import scala.collection.mutable.ArrayBuffer

/**
 * TODO
 *
 * @author dlwh
 **/
class ILPGrammarClusterer(maxNumPartitions: Int = 32, partitionBadnessThreshold: Int = 55) extends GrammarClusterer {
  /**
   * Solve an ILP that basically says:
   *
   * minimize total_badness
   * where total_badness = sum_partition badness_partition
   * badness_partition = num of variables over partitionBadnessThreshold we have to keep track of in each partition
   * numVariables_partition = sum_parents  present_as_parent_partition_parent + \sum_right present_as_right_partition_right  + \sum_left present_as_left_partition_left
   * \forall_parent \sum_partitions present_as_parent_partition_parent = 1
   * \forall_rules present_as_parent_partition_parent implies present_as_left_partition_left_child
   * \forall_rules present_as_parent_partition_parent implies present_as_right_partition_right_child
   * @param rules
   * @param targetLabel
   * @return
   **/

  def partition(rules: IndexedSeq[(BinaryRule[Int], Int)], targetLabel: TargetLabel): IndexedSeq[immutable.IndexedSeq[(BinaryRule[Int], Int)]] = {
    val dsl = new LinearProgram
    import dsl._
    val constraints = new ArrayBuffer[Constraint]()
    val targetVariables = Array.tabulate(maxNumPartitions)(i => rules.map(_._1).map(targetLabel.target).toSet.iterator.map{ (l:Int) => l -> Binary(s"parent_${i}_$l")}.toMap)

    // each parent must be in exactly one partition
    // also, break some symmetries
    val pindices = targetVariables.head.keys.toIndexedSeq
    for( pindex <- pindices) {
      val i = math.min(pindex, maxNumPartitions - 1)
      // each parent has to be in a partition <= it's index in pindex
      constraints += (0 to i).map(targetVariables(_)(pindices(i))).reduceLeft[Expression](_ + _) =:= 1
      // can't be in any of the bigger ones
      constraints ++= (i+1 until maxNumPartitions).map(targetVariables(_)(pindices(i)) =:= 0.0)
    }
    /*
    for( l <- targetVariables.head.keys) {
      val eachPartitionL = targetVariables.map(_(l)).reduceLeft[Expression](_ + _)
      constraints += eachPartitionL =:= 1
    }
    */
    val leftVariables = Array.tabulate(maxNumPartitions)(i => rules.map(_._1).flatMap(targetLabel.clusterPieces(_)._1).toSet.iterator.map{ (l:Int) => l -> Binary(s"left_${i}_$l")}.toMap)
    val rightVariables = Array.tabulate(maxNumPartitions)(i => rules.map(_._1).flatMap(targetLabel.clusterPieces(_)._2).toSet.iterator.map{ (l:Int) => l -> Binary(s"right_${i}_$l")}.toMap)

    val rulesByParent = rules.map(_._1).groupBy(targetLabel.target)

    // if a parent is in a partition, so must all left and right children in its rules
    for {
      (target, rules) <- rulesByParent
    } {
      val pindex = math.min(pindices.indexOf(target), maxNumPartitions - 1)
      val (lefts, rights) = rules.map(targetLabel.clusterPieces).unzip
      val left = lefts.reduceLeft(_ | _).toIndexedSeq
      val right = rights.reduceLeft(_ | _).toIndexedSeq
      for(p <- 0 to pindex) {
        for(l <- left)
          constraints += targetVariables(p)(target) <= leftVariables(p)(l)
        for(r <- right)
          constraints += targetVariables(p)(target) <= rightVariables(p)(r)
      }

      // follows from symmetry breaking above
      for(l <- left)
        constraints += (0 to pindex).map(p => leftVariables(p)(l)).reduceLeft[Expression](_ + _) >= 1.0
      for(l <- right)
        constraints += (0 to pindex).map(p => rightVariables(p)(l)).reduceLeft[Expression](_ + _) >= 1.0
    }

    val leftSums = leftVariables.map(_.values.reduceLeft[Expression](_ + _))
    val rightSums = rightVariables.map(_.values.reduceLeft[Expression](_ + _))
    val targetSums = targetVariables.map(_.values.reduceLeft[Expression](_ + _))
    val badnesses: Array[dsl.Real] = Array.tabulate(maxNumPartitions)(p => Real(s"badness_$p"))

    for(p <- 0 until maxNumPartitions) {
      constraints += badnesses(p) >= 0.0
      constraints += (leftSums(p) + rightSums(p) + targetSums(p) - partitionBadnessThreshold) <= badnesses(p)
    }



    val objective = -badnesses.reduceLeft[Expression](_ + _) subjectTo (constraints: _*)
    println(objective)
    val result = maximize(objective)
    val parentPartitions = targetVariables.map(partitionVariables => partitionVariables.collect{ case (k, v) if result.valueOf(v) >= 0.9 => k})
    val groupedRules = rules.groupBy( pair => targetLabel.target(pair._1))
    println(parentPartitions, result.value)
    parentPartitions.map(targetsInPartition => targetsInPartition.flatMap(groupedRules).toIndexedSeq).toIndexedSeq.filter(_.nonEmpty)
  }
}
