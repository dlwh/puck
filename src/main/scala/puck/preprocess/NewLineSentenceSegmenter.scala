package puck.preprocess


import java.text.BreakIterator
import java.util.Locale
import epic.trees.Span
import epic.slab._
import epic.slab.Sentence
import epic.preprocess.SentenceSegmenter
import java.util.regex.Pattern
import scala.collection.mutable.ArrayBuffer

/**
 * TODO move to chalk
 *
 * @author dlwh
 **/
class NewLineSentenceSegmenter(locale: Locale = Locale.getDefault) extends SentenceSegmenter {

  private val regex = Pattern.compile("\n+")

  override def apply[In](slab: StringSlab[In]): StringSlab[In with Sentence] = {
    val m = regex.matcher(slab.content)

    val spans = new ArrayBuffer[(Span, Sentence)]()

    var start = 0
    while(m.find()) {
      val end = m.end()
      if(end - start > 1)
        spans += (Span(start, end) -> Sentence())
      start = end
    }
    spans += Span(start, slab.content.length) -> Sentence()


    slab.++[Sentence](spans)
  }
}
