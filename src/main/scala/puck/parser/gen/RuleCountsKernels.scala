package puck.parser.gen

import virtualization.lms.common.{RangeOps, Base}
import trochee.kernels.{Global, KernelOps}
import epic.trees.BinaryRule

/**
 *
 * @author dlwh
trait RuleCountsKernels[L] extends ParserCommon[L] { self: Base with KernelOps with RangeOps =>

  lazy val ecountLeftTermBinaries =  (grammar.partitionsLeftTermRules.zipWithIndex map { case (p, i) => ecountLeftTerms(i, p)})
  lazy val ecountRightTermBinaries =  (grammar.partitionsRightTermRules.zipWithIndex map { case (p, i) => ecountRightTerms(i, p)})

  def ecountBothTerms:IndexedSeq[Kernel] = grammar.partitionsBothTermRules.zipWithIndex map {case (p, i) => _ecountBothTerms(i, p)}

  def _ecountBothTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]): Kernel =  kernel("ecount_both_term_binaries_"+id, { (ruleTarget: Rep[RuleTarget with Global],
                                                                                                                                    outsideBots: Rep[ParseChart with Global],
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
      val ruleAcc = accumulatorForRules(grammar.bothTermRules)

      val sentOffset = offsets(sentence)
      val lengthOff = lengthOffsets(sentence)
      val leftInside = insidePos(lengthOff, begin, gram)
      val rightInside = insidePos(lengthOff, (end-1), gram)
      val parentOutside = outsideBots(sentOffset, begin, end, gram)

      doECountBinaryUpdates(ruleAcc, parentOutside, leftInside, rightInside, rules, gram, partition)

      ruleTarget(???) += ruleAcc
    } else unit() // needed because scala is silly
  })



  def ecountLeftTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]) = kernel("ecount_left_term_binaries_" + id, { (ruleTarget: Rep[RuleTarget with Global],
                                                                                                                                outsideBots: Rep[ParseChart with Global],
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
      if(begin !== 0) { // && causes crashes i nthe compiler
        val ruleAcc = accumulatorForRules(grammar.leftTermRules)

        val sentOffset = offsets(sentence)
        val lengthOff = lengthOffsets(sentence)
        val rightInside = insideTops(sentOffset, begin, end, gram)
        val leftTermInside = insideTags(lengthOff, begin-1, gram)
        val parentOutside = outsideBots(sentOffset, begin-1, end, gram)

        doECountBinaryUpdates(ruleAcc, parentOutside, leftTermInside, rightInside, rules, gram, partition)

        ruleTarget(???) += ruleAcc
      }
    } else unit() // needed because scala is silly
  })

  def ecountRightTerms(id: Int, partition: IndexedSeq[(BinaryRule[Int], Int)]) = kernel("ecount_right_term_binaries_"+id, { ( ruleTarget: Rep[RuleTarget with Global],
                                                                                                                              outsideBots: Rep[ParseChart with Global],
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

    if (end < length) {
      val ruleAcc = accumulatorForRules(grammar.rightTermRules)

      val sentOffset = offsets(sentence)
      val lengthOff = lengthOffsets(sentence)
      val leftInside = insideTops(sentOffset, begin, end, gram)
      val rightTermInside = insideTags(lengthOff, end, gram)
      val parentOutside = outsideBots(sentOffset, begin, end+1, gram)

      doECountBinaryUpdates(ruleAcc, parentOutside, leftInside, rightTermInside, rules, gram, partition)
      ruleTarget(???)  += ruleAcc

    } else unit() // needed because scala is silly
  })




  def ecountUnaries = kernel("ecount_unaries", { ( ruleTarget: Rep[RuleTarget with Global],
                                                   outsideTops: Rep[ParseChart with Global],
                                                  insideBots: Rep[ParseChart with Global],
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
      val top = outsideTops(sentOffset, begin, end, gram)
      val bot = insideBots(sentOffset, begin, end, gram)

      doECountUnaryUpdates(out, top, bot, rules, gram)
      ruleTarget(???)  += out
    } else unit() // needed because scala is silly
  } )

  def ecountTermUnaries = kernel("ecount_term_unaries", { (ruleTarget: Rep[RuleTarget with Global],
                                                           outsideTops: Rep[ParseChart with Global],
                                                           insidePos: Rep[TermChart with Global],
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
      val top = outsideTops(sentOffset, begin, end, gram)
      val bot = insidePos(lengthOff, begin, gram)
      doECountUnaryUpdates(out, top, bot, rules, gram)
      ruleTarget(???)  += out
    } else unit() // needed because scala is silly
  })



  def ecountNonterms(partitionId: Int, rulePartition: IndexedSeq[(BinaryRule[Int], Int)]) = {
    kernel("ecount_nonterms_right"+partitionId, { (  ruleTarget: Rep[RuleTarget with Global],
                                                    outsideTops: Rep[ParseChart with Global],
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
        val out = accumulatorForRules(rulePartition)

        val sentOffset = offsets(sentence)

        for(split <- (begin + unit(1)) until end) {
          val parentOutside = outsideBots(sentOffset, begin, end, gram)
          val left = insideTops(sentOffset, begin, split, gram)
          val right = insideTops(sentOffset, split, end, gram)
          doECountBinaryUpdates(out, parentOutside, left, right, rules, gram)
          unit()
        }

        ruleTarget(sentOffset, begin, end, gram) += out
      } else unit() // needed because scala is silly
    })
  }


  protected def doECountBinaryUpdates(ruleAcc: Accumulator,
                                      parentOutside: ParseCell,
                                      leftInside: ParseCell,
                                      rightInside: ParseCell,
                                      rules: Rep[RuleCell],
                                      gram: Rep[Int],
                                      partition: IndexedSeq[(BinaryRule[Int], Int)]):Rep[Unit]

  protected def doECountUnaryUpdates(ruleAcc: Accumulator,
                                      parentOutside: ParseCell,
                                      childInside: ParseCell,
                                      rules: Rep[RuleCell],
                                      gram: Rep[Int],
                                      partition: IndexedSeq[(BinaryRule[Int], Int)]):Rep[Unit]


}
 */
