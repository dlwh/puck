package puck.parser

import virtualization.lms.common.{RangeOps, Base}
import trochee.kernels.KernelOps
import spire.implicits._
import spire.syntax._
import spire.math._
import trochee.basic.SpireOps

/**
 * 
 * @author dlwh
 */
trait InliningInsideKernels extends UniformLoopInsideKernels { self: Base with KernelOps with RangeOps with SpireOps =>


  protected def doInsideUnaryUpdates(top: Accumulator, bot: ParseCell, rulePartition: IndexedSeq[(UnaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    for( (parent, rr) <- rulePartition.groupBy(_._1.parent)) {
      for((r,id) <- rr) {
        val botScore = bot(r.child)
        top.mad(parent, botScore, rules.rules(id, gram))
      }


    }
  }

  protected def doInsideBinaryUpdates(out: Accumulator, left: ParseCell, right: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    for( (leftChild, rr) <- rulePartition.groupBy(_._1.leftChild)) {
      val leftScore = left(leftChild)
      for((rightChild,rrr) <- rr.groupBy(_._1.rightChild)) {
        val rightScore = right(rightChild)
        val joint = leftScore * rightScore
        for((r,id) <- rrr) {
         out.mad(r.parent, joint, rules.rules(id, gram))
        }
      }
    }

  }
}

trait UniformLoopInsideKernels extends InsideKernels { self: Base with KernelOps with RangeOps with SpireOps =>
  protected def doLeftInsideTermUpdates(out: Accumulator, leftTerm: ParseCell, right: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideBinaryUpdates(out, leftTerm, right, grammar.leftTermRules, rules, gram)
  }

  protected def doBothInsideTermUpdates(out: Accumulator, leftTerm: ParseCell, rightTerm: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideBinaryUpdates(out, leftTerm, rightTerm, grammar.bothTermRules, rules, gram)
  }

  protected def doRightInsideTermUpdates(out: Accumulator, left: ParseCell, rightTerm: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideBinaryUpdates(out, left, rightTerm, grammar.rightTermRules, rules, gram)
  }

  protected def doNTInsideRuleUpdates(out: Accumulator, left: ParseCell, right: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideBinaryUpdates(out, left, right, rulePartition, rules, gram)
  }

  protected def doInsideUnaries(top: Accumulator, bot: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideUnaryUpdates(top, bot, grammar.unaryRules, rules, gram)
  }

  protected def doInsideTermUnaries(top: Accumulator, bot: ParseCell, rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit] = {
    doInsideUnaryUpdates(top, bot, grammar.unaryTermRules, rules, gram)
  }

  protected def doInsideBinaryUpdates(out: Accumulator, left: ParseCell, right: ParseCell, rulePartition: IndexedSeq[(BinaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit]
  protected def doInsideUnaryUpdates(top: Accumulator, bot: ParseCell, rulePartition: IndexedSeq[(UnaryRule[Int], Int)], rules: Rep[RuleCell], gram: Rep[Int]): Rep[Unit]
}
