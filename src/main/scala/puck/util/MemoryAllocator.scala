package puck.util

import com.nativelibs4java.opencl.{CLBuffer, CLMem, CLContext}
import java.util
import org.bridj.{Pointer, PointerIO}
import java.util.concurrent.atomic.AtomicInteger

/**
 *
 *
 * @author dlwh
 */
class MemoryAllocator(val maxSize: Long)(implicit context: CLContext) { mem =>
  def this()(implicit context: CLContext) = this(context.getMaxMemAllocSize)(context)
  assert(context.getMaxMemAllocSize >= maxSize, s"Can't allocate $maxSize bytes on $context")

  def newBudget = new Budget(0)

  private var _allocatedSize: Long = 0

  private def canAllocateBytes(size: Long) = _allocatedSize + size < maxSize

  private[util] def externalAllocate[T:Manifest](kind: CLMem.Usage, numElems: Long): CLBuffer[T] = synchronized {
    val clss = implicitly[Manifest[T]].runtimeClass
    def elemSize: Long = PointerIO.getInstance(clss).getTargetSize()
    val byteSize: Long = elemSize * numElems
    require(canAllocateBytes(byteSize))
    _allocatedSize += byteSize
    context.createBuffer(kind, clss.asInstanceOf[Class[T]], numElems)
  }
  private[util] def externalRelease(size: Long) = synchronized { _allocatedSize -= size }


  class Budget(private var _allocatedSize: Long = 0) { budget =>

    private val nextId = new AtomicInteger(0)

    class Reservation[T] private[Budget] (val usage: CLMem.Usage, private var _numElements: Long, val clss: Class[_]) {
      private[MemoryAllocator] val id = nextId.getAndIncrement()

      def numElements: Long = _numElements
      def elemSize: Long = PointerIO.getInstance(clss).getTargetSize()
      def size = _numElements * elemSize

      def tryChange(newElemCount: Long): Boolean = mem.synchronized { budget.synchronized {
        val newSize = newElemCount * elemSize
        val delta: Long = newSize - size
        val canExtend: Boolean = _allocatedSize + delta <= maxSize
        if( canExtend) {
          _numElements = newElemCount
          _allocatedSize += delta
        }
        canExtend
      }}


      def released = budget.synchronized { reservations.containsKey(this) }


      def release() {
        budget.synchronized {
          if(!released) {
            reservations.remove(id)
            _allocatedSize -= size
          }
        }
      }

      protected override def finalize() {
        release()
      }
    }

    def reserve[T:Manifest](usage: CLMem.Usage, size: Long): Option[Reservation[T]] = synchronized {
      if(_allocatedSize + size <= mem.maxSize) {
        val res = new Reservation[T](usage, size, implicitly[Manifest[T]].runtimeClass)
        reservations.put(res.id, res)
        _allocatedSize += size
        Some(res)
      } else {
        None
      }
    }

    def allocate():Option[Allocation[this.type]] = mem.synchronized{budget.synchronized{
      if(canCurrentlyAllocate) {
        Some(new Allocation(this))
      } else {
        None
      }
    }}

    def canCurrentlyAllocate = _allocatedSize <= maxSize - mem._allocatedSize

    def canCurrentlySupportDelta[T:Manifest](elemDelta: Long) = {
      val elemSize = PointerIO.getInstance(implicitly[Manifest[_]].runtimeClass).getTargetSize()
      _allocatedSize + (elemDelta * elemSize) <= maxSize - mem._allocatedSize
    }

    private[MemoryAllocator] val reservations = new util.HashMap[Int, Reservation[_]]()
  }

  class Allocation[B<:Budget]private[MemoryAllocator] (val budget: B) {
    import scala.collection.JavaConverters._

    def getPointer[T](r: budget.Reservation[T]) = pointers(r.id).asInstanceOf[CLBuffer[T]]

    private val pointers = mem.synchronized {
      val ptrs = budget.synchronized { budget.reservations.asScala.mapValues(r => context.createBuffer(r.usage, r.clss, r.size)).toMap}
      mem._allocatedSize += totalSize
      ptrs
    }
    val totalSize: Long = pointers.values.map(b => b.getElementSize * b.getElementCount).sum

    def release() {
      synchronized  {
        if(!released) {
          actuallyRelease()
        }
      }
    }

    protected override def finalize() {
      if(!released) {
        actuallyRelease()
      }

    }


    private def actuallyRelease() {
      mem.synchronized {
        mem._allocatedSize -= totalSize
        pointers.values.foreach {
          _.release()
        }
        _released = true
      }
    }

    private var _released = false
    def released = _released

  }

}
