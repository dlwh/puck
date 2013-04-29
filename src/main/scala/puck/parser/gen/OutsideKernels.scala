package puck.parser.gen

import virtualization.lms.common.{RangeOps, Base}
import trochee.kernels.{Global, KernelOps}
import epic.trees.BinaryRule

/**
 *
 * @author dlwh
 */
trait OutsideKernels[L] extends ParserCommon[L] { self: Base with KernelOps with RangeOps =>

  lazy val outsideLeftTermBinaries =  (grammar.partitionsLeftTermRules.zipWithIndex map { case (p, i) => outsideLeftTerms(i, p)})
  lazy val outsideRightTermBinaries =  (grammar.partitionsRightTermRules.zipWithIndex map { case (p, i) => outsideRightTerms(i, p)})

  def outsideBothTerms:IndexedSeq[Kernel] = grammar.partitionsBothTermRules.zipWithIndex map {case (p, i) => _outsideBothTerms(i, p)}

  def _outsideBothTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]): Kernel =  kernel("outside_both_term_binaries_"+id, { (
                                                                                                                                    outsideBots: Rep[ParseChart with Global],
                                                                                                                                    outsidePos: Rep[TermChart with Global],
                                                                                                                                    insidePos: Rep[TermChart with Global],
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
      val leftOut = accumulatorForLeftChildren(grammar.bothTermRules)
      val rightOut = accumulatorForRightChildren(grammar.bothTermRules)

      val sentOffset = offsets(sentence)
      val lengthOff = lengthOffsets(sentence)
      val leftInside = insidePos(lengthOff, begin, gram)
      val rightInside = insidePos(lengthOff, (end-1), gram)
      val parentOutside = outsideBots(sentOffset, begin, end, gram)

      doBothOutsideUpdates(leftOut, rightOut, parentOutside, leftInside, rightInside, rules, gram, partition)

      outsidePos(lengthOff, begin, gram) += leftOut
      outsidePos(lengthOff, (end-1), gram) += rightOut
    } else unit() // needed because scala is silly
  })



  def outsideLeftTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]) = kernel("outside_left_term_binaries_" + id, { (outsideBots: Rep[ParseChart with Global],
                                                                                                                               outsideTags: Rep[TermChart with Global],
                                                                                                                               outsideTops: Rep[ParseChart with Global],
                                                                                                                             insideTops: Rep[ParseChart with Global],
                                                                                                                             insideTags: Rep[TermChart with Global],
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
      val leftOut = accumulatorForLeftChildren(grammar.leftTermRules)
      val rightOut = accumulatorForRightChildren(grammar.leftTermRules)

      val sentOffset = offsets(sentence)
      val lengthOff = lengthOffsets(sentence)
      val rightInside = insideTops(sentOffset, begin+1, end, gram)
      val leftTermInside = insideTags(lengthOff, begin, gram)
      val parentOutside = outsideBots(sentOffset, begin, end, gram)

      doBothOutsideUpdates(leftOut, rightOut, parentOutside, leftTermInside, rightInside, rules, gram, partition)

      outsideTops(sentOffset, begin+1, end, gram)  += rightOut
      outsideTags(lengthOff, begin, gram) += leftOut
    } else unit() // needed because scala is silly
  })

  def outsideRightTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]) = kernel("outside_right_term_binaries_"+id, { (outsideBots: Rep[ParseChart with Global],
                                                                                                                               outsideTags: Rep[TermChart with Global],
                                                                                                                               outsideTops: Rep[ParseChart with Global],
                                                                                                                             insideTops: Rep[ParseChart with Global],
                                                                                                                             insideTags: Rep[TermChart with Global],
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
      val leftOut = accumulatorForLeftChildren(grammar.rightTermRules)
      val rightOut = accumulatorForRightChildren(grammar.rightTermRules)

      val sentOffset = offsets(sentence)
      val lengthOff = lengthOffsets(sentence)
      val leftInside = insideTops(sentOffset, begin, end-1, gram)
      val rightTermInside = insideTags(lengthOff, end-1, gram)
      val parentOutside = outsideBots(sentOffset, begin, end, gram)

      doBothOutsideUpdates(leftOut, rightOut, parentOutside, leftInside, rightTermInside, rules, gram, partition)
      outsideTops(sentOffset, begin, end-1, gram)  += leftOut
      outsideTags(lengthOff, end-1, gram) += rightOut

    } else unit() // needed because scala is silly
  })




  def outsideUnaries = kernel("outside_unaries", { (outsideTops: Rep[ParseChart with Global],
                                                  outsideBots: Rep[ParseChart with Global],
                                                  offsets: Rep[Array[Int] with Global],
                                                  lengths: Rep[Array[Int] with Global],
                                                  spanLength: Rep[Int],
                                                  rules: Rep[RuleCell with Global]) =>
    val sentence = globalId(0)
    val begin = globalId(1)
    val gram = globalId(2)
    val end = begin + spanLength
    val length = lengths(sentence)
    if(spanLength === length) {
      outsideTops(offsets(sentence), 0, length, gram)(grammar.root) = one
    }


    if (end <= length) {

      val sentOffset = offsets(sentence)
      val out = accumulatorForChildren(grammar.unaryRules)
      val top = outsideTops(sentOffset, begin, end, gram)

      doOutsideUnaries(out, top, rules, gram)
      outsideBots(sentOffset, begin, end, gram) += out
    } else unit() // needed because scala is silly
  } )

  def outsideTermUnaries = kernel("outside_term_unaries", { (outsideTops: Rep[ParseChart with Global],
                                                           outsidePos: Rep[TermChart with Global],
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
      val out = accumulatorForChildren(grammar.unaryTermRules)
      val parent = outsideTops(sentOffset, begin, end, gram)
      doOutsideTermUnaries(out, parent, rules, gram)
      outsidePos(lengthOff, begin, gram) += out
    } else unit() // needed because scala is silly
  })



  def outsideNontermsRight(partitionId: Int, rulePartition: IndexedSeq[(BinaryRule[Int], Int)]) = {
    kernel("outside_nonterms_right"+partitionId, { ( outsideTops: Rep[ParseChart with Global],
                                                     outsideBots: Rep[ParseChart with Global],
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

      if ( end <= length) {
        val out = accumulatorForRightChildren(rulePartition)

        val sentOffset = offsets(sentence)

        for(completion <- 0 until begin) {
          val parent = outsideBots(sentOffset, completion, end, gram)
          val left = insideTops(sentOffset, completion, begin, gram)
          doNTOutsideRuleRightUpdates(out, parent, left, rulePartition, rules, gram)

          unit()
        }

        outsideTops(sentOffset, begin, end, gram) += out
      } else unit() // needed because scala is silly
    })
  }

  def outsideNontermsLeft(partitionId: Int, rulePartition: IndexedSeq[(BinaryRule[Int], Int)]) = {
    kernel("outside_nonterms_left"+partitionId, { (outsideTops: Rep[ParseChart with Global],
                                                    outsideBots: Rep[ParseChart with Global],
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

      if ( end <= length) {
        val out = accumulatorForLeftChildren(rulePartition)

        val sentOffset = offsets(sentence)

        for(completion <- (end+unit(1)) until (length + unit(1))) {
          val parent = outsideBots(sentOffset, begin, completion, gram)
          val right = insideTops(sentOffset, end, completion, gram)
          doNTOutsideRuleLeftUpdates(out, parent, right, rulePartition, rules, gram)

          unit()
        }

        outsideTops(sentOffset, begin, end, gram) += out
      } else unit() // needed because scala is silly
    })
  }


  protected def doBothOutsideUpdates(leftOut: Accumulator,
                                     rightOut: Accumulator,
                                     outsideParent: ParseCell,
                                     leftTerm: ParseCell,
                                     rightTerm: ParseCell,
                                     rules: Rep[RuleCell],
                                     gram: Rep[Int],
                                     partition: IndexedSeq[(BinaryRule[Int], Int)]):Rep[Unit]

  protected def doNTOutsideRuleRightUpdates(out: Accumulator,
                                            parent: ParseCell,
                                            leftInside: ParseCell,
                                            rulePartition: IndexedSeq[(BinaryRule[Int], Int)],
                                            rules: Rep[RuleCell],
                                            gram: Rep[Int]):Rep[Unit]

  protected def doNTOutsideRuleLeftUpdates(out: Accumulator,
                                           parent: ParseCell,
                                           rightInside: ParseCell,
                                           rulePartition: IndexedSeq[(BinaryRule[Int], Int)],
                                           rules: Rep[RuleCell],
                                           gram: Rep[Int]):Rep[Unit]


  protected def doOutsideUnaries(top: Accumulator,
                                 bot: ParseCell,
                                 rules: Rep[RuleCell],
                                 gram: Rep[Int]):Rep[Unit]

  protected def doOutsideTermUnaries(top: Accumulator,
                                     bot: ParseCell,
                                     rules: Rep[RuleCell],
                                     gram: Rep[Int]):Rep[Unit]
}
