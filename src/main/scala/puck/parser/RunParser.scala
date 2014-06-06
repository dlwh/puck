package puck
package parser

import epic.trees.annotations.{Xbarize, TreeAnnotator}
import epic.trees._
import java.io._
import breeze.config.CommandLineParser
import java.util.zip.ZipFile
import com.nativelibs4java.opencl.{JavaCL, CLContext}
import java.util.{Comparator, Collections}
import scala.collection.mutable.ArrayBuffer
import com.typesafe.scalalogging.slf4j.LazyLogging
import puck.{BatchFunctionAnnotatorService, AnnotatorService}
import scala.io.{Source, Codec}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong, AtomicInteger}
import scala.concurrent.{Future, ExecutionContext}
import java.util.concurrent.{PriorityBlockingQueue, ConcurrentLinkedQueue, TimeUnit}
import scala.concurrent.duration.Duration
import epic.trees.Debinarizer.AnnotatedLabelDebinarizer
import epic.preprocess.{TreebankTokenizer, StreamSentenceSegmenter, NewLineSentenceSegmenter, MLSentenceSegmenter}
import chalk.text.LanguagePack
import chalk.text.tokenize.WhitespaceTokenizer
//import epic.util.FIFOWorkQueue
import scala.concurrent.Await
import scala.util.{Failure, Success}

/**
 * TODO
 *
 * @author dlwh
 */
object RunParser extends LazyLogging {
  import ExecutionContext.Implicits.global

  case class Params(device: String = "nvidia",
                    sentences: String = "trained",
                    tokens: String = "default",
                    profile: Boolean = false,
                    numToParse: Int = 1000,
                    grammar: File = new File("grammar.grz"),
                    maxParseLength: Int = 10000,
                    mem: String = "3g",
                    maxLength: Int = 50,
                    outputSuffix: String = ".parsed")

  def main(args: Array[String]) {
    val (config, files) = CommandLineParser.parseArguments(args)
    val params:Params = config.readIn[Params]("")
    import params._

    implicit val context: CLContext = {
      val (good, bad) = JavaCL.listPlatforms().flatMap(_.listAllDevices(true)).partition(d => params.device.r.findFirstIn(d.toString.toLowerCase()).nonEmpty)
      if(good.isEmpty) {
        JavaCL.createContext(Collections.emptyMap(), bad.sortBy(d => d.toString.toLowerCase.contains("geforce")).last)
      } else {
        JavaCL.createContext(Collections.emptyMap(), good.head)
      }

    }
    logger.info(s"Using context $context")

    val parserData = CLParserData.readSequence[AnnotatedLabel, AnnotatedLabel, String](new ZipFile(params.grammar))

    val sentenceSegmenter = {
      val base = params.sentences.toLowerCase match {
        case "java" => LanguagePack.English.sentenceSegmenter
        case "default" | "trained" => MLSentenceSegmenter.bundled().get
        case "newline" => new NewLineSentenceSegmenter()
      }
      new StreamSentenceSegmenter(base)
    }
    val tokenizer = params.tokens.toLowerCase match {
      case "default" | "treebank" => new TreebankTokenizer
      case "none" | "whitespace" => new WhitespaceTokenizer
    }

    val parser = new CLParser(parserData, CLParser.parseMemString(mem), profile = profile)

    val service = AnnotatorService.fromBatchFunction(parser.parse(_:IndexedSeq[IndexedSeq[String]]))

    logger.info("Up and running")

    val iter = if(files.length == 0) Iterator(System.in -> System.out) else files.iterator.map(f => new BufferedInputStream(new FileInputStream(f)) -> new PrintStream(new BufferedOutputStream(new FileOutputStream(new File(f.toString + params.outputSuffix)))))


    var producedIndex = 0L
    val timeIn = System.currentTimeMillis()

    val successFutures = ArrayBuffer[Future[Any]]()

    for( (src, sink) <- iter) {
      val queue = FIFOWorkQueue(sentenceSegmenter.sentences(src).filter(_.trim.nonEmpty)){sent =>
        val words = tokenizer(sent).toIndexedSeq
        assert(words.length > 0)

        producedIndex += 1
        if(words.length < maxLength) {
          service(words).map {
            case Success(t) => t.map(_.label).render(words, newline = false)
            case Failure(ex) => ex.printStackTrace(); "(())"
          }
        } else {
          Future.successful("(())")
        }


      }
      successFutures += Future {
        queue.foreach { q => sink.println(Await.result(q, Duration.Inf)) }
      }
      successFutures.last.onComplete{_ => src.close();sink.close()}
      //      Await.result(successFutures.last, Duration.Inf)

    }
    service.flush()

    Await.result(Future.sequence(successFutures), Duration.Inf)

    val timeOut = System.currentTimeMillis()
    val wallTime = (timeOut - timeIn) / 1E3
    logger.info(f"Parsing took ${wallTime}s seconds. ${producedIndex/wallTime}%.3f sentences per second.")



  }

}
