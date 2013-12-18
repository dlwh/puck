package puck.cluster

import breeze.math.{MutableInnerProductSpace, MutablizingAdaptor, InnerProductSpace}
import breeze.math.MutablizingAdaptor.InnerProductSpaceAdaptor
import breeze.stats.distributions.{Multinomial, RandBasis, Rand}
import epic.util.{Optional, NotProvided}
import scala.collection.mutable.ArrayBuffer
import breeze.linalg._
import breeze.util.Implicits._
import breeze.util._
import breeze.numerics._





class BalancedKMeans[T](k: Int, tolerance: Double = 1E-4, distance: Optional[(T,T)=>Double] = NotProvided)(implicit random: RandBasis = Rand, innerProductSpace: InnerProductSpace[T, Double]) extends Clusterer[T] {
  private val ensured = MutablizingAdaptor.ensureMutable(innerProductSpace)
  import ensured.{wrap, unwrap}
  import ensured.Wrapper
  import ensured.mutaVspace._
  private val metric = distance.getOrElse( (a:T,b:T) => math.pow(norm(wrap(a) - wrap(b)), 2))

  case class State(means: IndexedSeq[T], clusters: IndexedSeq[IndexedSeq[T]], error: Double, previousError: Double=Double.PositiveInfinity) {
    def converged = previousError <= error * (1.0001)
  }


  def cluster(points: IndexedSeq[T]): IndexedSeq[IndexedSeq[T]] = {
    val lastState = iterates(points).last
    lastState.clusters
  }

  def iterates(points: IndexedSeq[T]) = {
    if(points.length <= k) Iterator(State(points, points.map(IndexedSeq(_)), 0.0))
    else {
      Iterator.iterate(initialState(points)){ current =>
        val distances = points.par.map { x =>
          val clusterSizes = if(current.clusters eq null) Array.fill(current.means.size)(1.0/k) else current.clusters.map(_.size.toDouble / points.length).toArray
          val distances = (current.means.zipWithIndex).map{ case(m,i) => (clusterSizes(i) + 1) * metric(x,m) - math.log(clusterSizes(i)+1)}
          val minPoint = distances.argmin // which cluster minimizes euclidean distance
          (x, minPoint,distances(minPoint))
        }

      println(distances.map(_._2))
        val newClusters = distances.groupBy(_._2).seq.values.map{_.map(_._1)}
      println(newClusters.map(_.size))

        val newMeans = newClusters.par.map(clust => clust.foldLeft(zeros(wrap(clust.head)))(_ += wrap(_)) /= clust.size.toDouble)
        val error = distances.map(tuple =>tuple._3).sum

        println(current.error, error)

        State(newMeans.map(unwrap).seq.toIndexedSeq, newClusters.map(_.toIndexedSeq).toIndexedSeq, error, current.error)
      }.takeUpToWhere(state => state.converged)
    }


  }

  private def initialState(points: IndexedSeq[T]):State = {
    val probs = Array.fill(points.length)(1.0)
    val means = new ArrayBuffer[T]()

    while(means.length < k) {
      val newMean = points(Multinomial(new DenseVector(probs)).sample())
      means += newMean
      for(i <- (0 until probs.length).par) {
        probs(i) = math.min(probs(i), metric(newMean, points(i)))
      }
    }

    new State(means, null, 1E200)
  }
}
