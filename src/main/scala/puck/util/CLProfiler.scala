package puck.util
import com.nativelibs4java.opencl._

class CLProfiler(name: String) {
  private var startingWallTime:Long = -1L
  private var totalWallTime: Long = 0
  private var events = collection.mutable.ArrayBuffer[CLEvent]()

  def tick() {
    startingWallTime = System.currentTimeMillis
  }

  def tock() = {
    assert(startingWallTime != -1L)
    val timeOut = System.currentTimeMillis
    totalWallTime += (timeOut - startingWallTime)
    startingWallTime = -1L
  }

  def +=(event: CLEvent):this.type = {
    if (event ne null) events += event 
    this
  }

  def ++=(event: Traversable[CLEvent]):this.type = {
    if (event ne null) events ++= event 
    this
  }

  def adding(events: IndexedSeq[CLEvent]):events.type = {
    this.events ++= events
    events
  }

  def adding(event: CLEvent):event.type = {
    this.events += event
    event
  }



  def clear() {
    startingWallTime = -1L
    totalWallTime = -1L
    events.clear()
  }

  override def toString() = {
    val eventTimes = events.filter(_ ne null).map(e => (e.getProfilingCommandEnd - e.getProfilingCommandStart)/1E9).sum
    val queueTimes = events.filter(_ ne null).map(e => (e.getProfilingCommandStart - e.getProfilingCommandQueued)/1E9).sum
    val submitTimes = events.filter(_ ne null).map(e => (e.getProfilingCommandEnd - e.getProfilingCommandSubmit)/1E9).sum
    f"Profile $name: ${totalWallTime / 1000.}%.3fs wall time. ${eventTimes}%.6fs processing time. ${queueTimes}%.6fs in queue. ${submitTimes}%.6fs submit."
  }

}
