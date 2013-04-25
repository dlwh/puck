package puck.parser.gen

import virtualization.lms.common.{RangeOps, Base}
import trochee.kernels.{Constant, Global, KernelOps}
import spire.algebra._
import spire.implicits._
import spire.syntax._
import spire.math._
import epic.trees.BinaryRule

/**
 *
 * @author dlwh
 */
trait InsideKernels[L] extends ParserCommon[L] { self: Base with KernelOps with RangeOps =>

  def insideTermBinaries:IndexedSeq[Kernel] = (
    (grammar.partitionsLeftTermRules.zipWithIndex map { case (p, i) => insideLeftTerms(i, p)})
      ++ (grammar.partitionsRightTermRules.zipWithIndex map { case (p, i) => insideRightTerms(i, p)})
    )

  def insideBothTerms:IndexedSeq[Kernel] = grammar.partitionsBothTermRules.zipWithIndex map {case (p, i) => _insideBothTerms(i, p)}

  def _insideBothTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]): Kernel =  kernel6("inside_both_term_binaries_"+id){ (insideBots: Rep[ParseChart with Global],
                                                                                                                                    posTags: Rep[TermChart with Global],
                                                                                                                                    offsets: Rep[Array[Int] with Global],
                                                                                                                                    lengths: Rep[Array[Int] with Global],
                                                                                                                                    lengthOffsets: Rep[Array[Int] with Global],
                                                                                                                                    rules: Rep[RuleCell with Global]) =>

    val spanLength = 2
    val sentence = globalId(0)
    val begin = globalId(1)
    val gram = globalId(2)
    val end = begin + spanLength
    val length = lengths(sentence)

    if (end <= length) {
      val out = accumulatorForRules(grammar.bothTermRules)

      val sentOffset = offsets(sentence)
      val lengthOff = lengthOffsets(sentence)
      val leftTerm = posTags(lengthOff, begin, gram)
      val rightTerm = posTags(lengthOff, (end-1), gram)
      doBothInsideTermUpdates(out, leftTerm, rightTerm, rules, gram, partition)

      insideBots(sentOffset, begin, end, gram) = out
    } else unit() // needed because scala is silly
  }



  def insideLeftTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]) = kernel8("inside_left_term_binaries_" + id){ (insideBots: Rep[ParseChart with Global],
                                                                                                                             insideTops: Rep[ParseChart with Global],
                                                                                                                             posTags: Rep[TermChart with Global],
                                                                                                                             offsets: Rep[Array[Int] with Global],
                                                                                                                             lengths: Rep[Array[Int] with Global],
                                                                                                                             lengthOffsets: Rep[Array[Int] with Global],
                                                                                                                             spanLength: Rep[Int],
                                                                                                                             rules: Rep[RuleCell with Global]) =>
    val sentence = globalId(0)
    val begin = globalId(1)
    val gram = globalId(2)
    val end = begin + spanLength
    val length = lengths(sentence)

    if (end <= length) {
      val out = accumulatorForRules(partition)

      val sentOffset = offsets(sentence)
      val lengthOff = lengthOffsets(sentence)
      val right = insideTops(sentOffset, begin+1, end, gram)
      val leftTerm = posTags(lengthOff, begin, gram)

      doLeftInsideTermUpdates(out, leftTerm, right, rules, gram, partition)
      insideBots(sentOffset, begin, end, gram) = out

    } else unit() // needed because scala is silly
  }

  def insideRightTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]) = kernel8("inside_right_term_binaries_"+id){ (insideBots: Rep[ParseChart with Global],
                                                                                                                             insideTops: Rep[ParseChart with Global],
                                                                                                                             posTags: Rep[TermChart with Global],
                                                                                                                             offsets: Rep[Array[Int] with Global],
                                                                                                                             lengths: Rep[Array[Int] with Global],
                                                                                                                             lengthOffsets: Rep[Array[Int] with Global],
                                                                                                                             spanLength: Rep[Int],
                                                                                                                             rules: Rep[RuleCell with Global]) =>
    val sentence = globalId(0)
    val begin = globalId(1)
    val gram = globalId(2)
    val end = begin + spanLength
    val length = lengths(sentence)

    if (end <= length) {
      val out = accumulatorForRules(partition)

      val sentOffset = offsets(sentence)
      val lengthOff = lengthOffsets(sentence)
      val left = insideTops(sentOffset, begin, end-1, gram)
      val rightTerm = posTags(lengthOff, end-1, gram)

      doRightInsideTermUpdates(out, left, rightTerm, rules, gram, partition)
      insideBots(sentOffset, begin, end, gram) = out

    } else unit() // needed because scala is silly
  }


  protected def doLeftInsideTermUpdates(out: Accumulator,
                                        leftTerm: ParseCell,
                                        right: ParseCell,
                                        rules: Rep[RuleCell],
                                        gram: Rep[Int],
                                        partition: IndexedSeq[(BinaryRule[Int], Int)]):Rep[Unit]


  protected def doBothInsideTermUpdates(out: Accumulator,
                                        leftTerm: ParseCell,
                                        rightTerm: ParseCell,
                                        rules: Rep[RuleCell],
                                        gram: Rep[Int],
                                        partition: IndexedSeq[(BinaryRule[Int], Int)]):Rep[Unit]

  protected def doRightInsideTermUpdates(out: Accumulator,
                                         left: ParseCell,
                                         rightTerm: ParseCell,
                                         rules: Rep[RuleCell],
                                         gram: Rep[Int],
                                         partition: IndexedSeq[(BinaryRule[Int], Int)]):Rep[Unit]

  protected def doNTInsideRuleUpdates(out: Accumulator,
                                      left: ParseCell,
                                      right: ParseCell,
                                      rulePartition: IndexedSeq[(BinaryRule[Int], Int)],
                                      rules: Rep[RuleCell],
                                      gram: Rep[Int]):Rep[Unit]


  def insideUnaries = kernel6("inside_unaries"){ (insideBots: Rep[ParseChart with Global],
                                                  insideTops: Rep[ParseChart with Global],
                                                  offsets: Rep[Array[Int] with Global],
                                                  lengths: Rep[Array[Int] with Global],
                                                  spanLength: Rep[Int],
                                                  rules: Rep[RuleCell with Global]) =>
    val sentence = globalId(0)
    val begin = globalId(1)
    val gram = globalId(2)
    val end = begin + spanLength
    val length = lengths(sentence)

    if (end <= length) {
      val sentOffset = offsets(sentence)
      val out = accumulatorForRules(grammar.unaryRules)
      val bot = insideBots(sentOffset, begin, end, gram)
      doInsideUnaries(out, bot, rules, gram)
      insideTops(sentOffset, begin, end, gram) = out
    } else unit() // needed because scala is silly
  }

  def insideTermUnaries = kernel6("inside_term_unaries"){ (insidePos: Rep[TermChart with Global],
                                                           insideTops: Rep[ParseChart with Global],
                                                           offsets: Rep[Array[Int] with Global],
                                                           lengths: Rep[Array[Int] with Global],
                                                           lengthOffsets: Rep[Array[Int] with Global],
                                                           rules: Rep[RuleCell with Global]) =>
    val sentence = globalId(0)
    val begin = globalId(1)
    val gram = globalId(2)
    val end = begin + 1
    val length = lengths(sentence)
    val lengthOff = lengthOffsets(sentence)

    if (end <= length) {
      val sentOffset = offsets(sentence)
      val out = accumulatorForRules(grammar.unaryTermRules)
      val bot = insidePos(lengthOff, begin, gram)
      doInsideTermUnaries(out, bot, rules, gram)
      insideTops(sentOffset, begin, end, gram) = out
    } else unit() // needed because scala is silly
  }

  protected def doInsideUnaries(top: Accumulator,
                                bot: ParseCell,
                                rules: Rep[RuleCell],
                                gram: Rep[Int]):Rep[Unit]

  protected def doInsideTermUnaries(top: Accumulator,
                                    bot: ParseCell,
                                    rules: Rep[RuleCell],
                                    gram: Rep[Int]):Rep[Unit]


  def insideNonterms(partitionId: Int, rulePartition: IndexedSeq[(BinaryRule[Int], Int)]) = {
    kernel6("inside_nonterms_"+partitionId){ (insideBots: Rep[ParseChart with Global],
                                              insideTops: Rep[ParseChart with Global],
                                              offsets: Rep[Array[Int] with Global],
                                              lengths: Rep[Array[Int] with Global],
                                              spanLength: Rep[Int],
                                              rules: Rep[RuleCell with Global]) =>
      val sentence = globalId(0)
      val begin = globalId(1)
      val gram = globalId(2)
      val end = begin + spanLength
      val length = lengths(sentence)

      if (end <= length) {
        val out = accumulatorForRules(rulePartition)

        val sentOffset = offsets(sentence)

        for(split <- (begin + unit(1)) until end) {
          val left = insideTops(sentOffset, begin, split, gram)
          val right = insideTops(sentOffset, split, end, gram)
          doNTInsideRuleUpdates(out, left, right, rulePartition, rules, gram)

          unit()
        }

        insideBots(sentOffset, begin, end, gram) = out
      } else unit() // needed because scala is silly
    }
  }


}
