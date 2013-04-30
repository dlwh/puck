package puck.parser.gen

import epic.trees._
import virtualization.lms.common.{RangeOps, Base}
import trochee.kernels.KernelOps
import trochee.basic.SpireOps

/**
 * 
 * @author dlwh
 */
trait InliningOutsideKernels[L] extends OutsideKernels[L] { self: Base with KernelOps with RangeOps with SpireOps =>


  protected def doBothOutsideUpdates(leftOut: Accumulator, rightOut: Accumulator, outsideParent: ParseCell, leftInside: ParseCell, rightInside: ParseCell, rules: Rep[RuleCell], gram: Rep[Int], partition: IndexedSeq[(BinaryRule[Int], Int)]): Rep[Unit] = {
    val leftCache = collection.mutable.Map[Int, Rep[Real]]()
    val rightCache = collection.mutable.Map[Int, Rep[Real]]()
    for ( (p, rs) <- partition.groupBy(_._1.parent)) {
      val pscore = outsideParent(p)
      for( (left, rrs) <- rs.groupBy(_._1.left)) {
        val lscore = leftCache.getOrElseUpdate(left, leftInside(left))
        val leftJoint = pscore * lscore
        for( (r,id) <- rrs) {
          val right = r.right
          val rscore = rightCache.getOrElseUpdate(right, rightInside(right))
          val rightJoint = pscore * rscore
          val ruleScore = rules.rules(id, gram)
          if(lscore !== Float.NegativeInfinity)
            expprintf("??? %d %d %f %f %f %f\\n", left, right, rightJoint, leftJoint, lscore, rscore)
          if(rscore !== Float.NegativeInfinity)
            expprintf("??? %d %d %f %f %f %f\\n", left, right, rightJoint, leftJoint, lscore, rscore)
          leftOut.mad(left, rightJoint, ruleScore)
          rightOut.mad(right, leftJoint, ruleScore)

        }
      }
    }
  }

  protected def doNTOutsideRuleRightUpdates(out: Accumulator, parent: ParseCell, leftInside: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    val leftCache = collection.mutable.Map[Int, Rep[Real]]()
    for ( (p, rs) <- rulePartition.groupBy(_._1.parent)) {
      val pscore = parent(p)
      for( (left, rrs) <- rs.groupBy(_._1.left)) {
        val lscore = leftCache.getOrElseUpdate(left, leftInside(left))
        val leftJoint = pscore * lscore
        for( (r,id) <- rrs) {
          val right = r.right
          val ruleScore = rules.rules(id, gram)
          out.mad(right, leftJoint, ruleScore)
        }
      }
    }
  }

  protected def doNTOutsideRuleLeftUpdates(out: Accumulator, parent: ParseCell, rightInside: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    val rightCache = collection.mutable.Map[Int, Rep[Real]]()
    for ( (p, rs) <- rulePartition.groupBy(_._1.parent)) {
      val pscore = parent(p)
      for( (right, rrs) <- rs.groupBy(_._1.right)) {
        val rscore = rightCache.getOrElseUpdate(right, rightInside(right))
        val rightJoint = pscore * rscore
        for( (r,id) <- rrs) {
          val left = r.left
          val ruleScore = rules.rules(id, gram)
          out.mad(left, rightJoint, ruleScore)
        }
      }
    }
  }

  protected def doOutsideUnaries(bot: Accumulator, top: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doOutsideUnaryUpdates(bot, top, grammar.unaryRules, rules, gram)
  }

  protected def doOutsideTermUnaries(bot: Accumulator, top: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doOutsideUnaryUpdates(bot, top, grammar.unaryTermRules, rules, gram)
  }

  protected def doOutsideUnaryUpdates(top: Accumulator, bot: ParseCell, rulePartition: IndexedSeq[(UnaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    for( (parent, rr) <- rulePartition.groupBy(_._1.parent)) {
      val parentScore = bot(parent)
      for((r,id) <- rr) {
        top.mad(r.child, parentScore, rules.rules(id, gram))
      }
    }
  }

  protected def doOutsideBinaryUpdates(out: Accumulator, left: ParseCell, right: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    for( (lc, rr) <- rulePartition.groupBy(_._1.left)) {
      val leftScore = left(lc)
      for((rc,rrr) <- rr.groupBy(_._1.right)) {
        val rightScore = right(rc)
        val joint = leftScore * rightScore
        for((r,id) <- rrr) {
         out.mad(r.parent, joint, rules.rules(id, gram))
        }
      }
    }

  }
}


