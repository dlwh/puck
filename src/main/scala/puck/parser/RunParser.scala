package puck.parser

import epic.trees.annotations.{Xbarize, TreeAnnotator}
import epic.trees._
import java.io.File
import breeze.config.CommandLineParser
import java.util.zip.ZipFile
import com.nativelibs4java.opencl.{JavaCL, CLContext}
import java.util.{Comparator, Collections}
import com.typesafe.scalalogging.slf4j.LazyLogging
import breeze.optimize.BatchDiffFunction
import puck.{BatchFunctionAnnotatorService, AnnotatorService}
import scala.io.{Source, Codec}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicLong, AtomicInteger}
import scala.concurrent.{Future, ExecutionContext}
import java.util.concurrent.{PriorityBlockingQueue, ConcurrentLinkedQueue, TimeUnit}
import scala.concurrent.duration.Duration

/**
 * TODO
 *
 * @author dlwh
 */
object RunParser extends LazyLogging {
  import ExecutionContext.Implicits.global

  case class Params(device: String = "nvidia",
                    profile: Boolean = false,
                    numToParse: Int = 1000,
                    grammar: File = new File("grammar.grz"),
                    maxParseLength: Int = 10000,
                    mem: String = "3g",
                    maxLength: Int = 50)

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

    val parser = new CLParser(parserData, CLParser.parseMemString(mem), profile = profile)

    val service = AnnotatorService.fromBatchFunction(parser.parse(_:IndexedSeq[IndexedSeq[String]]), flushInterval = Duration(100, TimeUnit.MILLISECONDS))

    logger.info("Up and running")

    val fileIter = if(files.nonEmpty) files.iterator.map(Source.fromFile(_)(Codec.UTF8)) else Iterator(Source.fromInputStream(System.in))

    var producedIndex = 0L
    val timeIn = System.currentTimeMillis()

    val output = new PriorityBlockingQueue[(String, Long)](100, new Comparator[(String, Long)] {
      override def compare(o1: (String, Long), o2: (String, Long)): Int = math.signum(o1._2 - o2._2).toInt
    })

    val stop = new AtomicBoolean(false)
    val consumedIndex = new AtomicLong()

    val consumer = new Thread(new Runnable {
      override def run(): Unit = {
        output.synchronized {
          while(!stop.get()) {
            output.wait()
            var drain = true
            while(drain) {
              output.peek() match {
                case null =>
                  drain = false
                case (str, prio) =>
                  if(prio == consumedIndex.get) {
                    drain = true
                    val x = output.poll
                    assert(x._2 == prio)
                    println(str)
                    val res = consumedIndex.incrementAndGet()
                    if(res == producedIndex) consumedIndex.synchronized { consumedIndex.notifyAll() }
                  } else {
                    drain = false
                  }
              }
            }

          }
        }
      }
    })
    consumer.start()

    for(f <- fileIter; line <- f.getLines()) {
      val words = line.trim.split(" ")
      val i = producedIndex
      producedIndex += 1
      if(words.length < maxLength) {
        service(words).foreach { guess =>
          val tree: Tree[String] = AnnotatedLabelChainReplacer.replaceUnaries(guess).map(_.label)
          val guessTree = Trees.debinarize(Trees.deannotate(tree))
          val rendered = guessTree.render(words, newline = false)
          output.add(rendered -> i)
          output.synchronized {
            output.notifyAll()
          }
        }
      } else {
        output.add("(())" -> i)
        output.synchronized {
          output.notifyAll()
        }
      }
    }
    service.flush()

    consumedIndex.synchronized {
      while (consumedIndex.get() != producedIndex) {
        consumedIndex.wait()
      }
    }

    stop.set(true)
    output.synchronized {
      output.notifyAll()
    }

    val timeOut = System.currentTimeMillis()
    val wallTime = (timeOut - timeIn) / 1E3
    logger.info(f"Parsing took ${wallTime}s seconds. ${producedIndex/wallTime}%.3f sentences per second.")



  }

}
