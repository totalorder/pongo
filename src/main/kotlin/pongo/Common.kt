package pongo

import org.opencv.core.CvType
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import java.util.ArrayList

data class Contour(val index: Int, val matOfPoint: MatOfPoint) {
    fun area(): Double  = Imgproc.contourArea(matOfPoint)
    fun getPoint(index: Int): Point {
        return matOfPoint.toArray()[index]
    }
}

data class Rect(private val unsortedPoints: List<Point>) {
    val points = sortedPoints(unsortedPoints)

    private fun sortedPoints(unsortedPoints: List<Point>): List<Point> {
        val ltr = unsortedPoints.sortedBy { it.x }
        val ltrttb = ltr.take(2).sortedBy { it.y }
        val rtlttb = ltr.takeLast(2).sortedBy { it.y }
        val topLeft = ltrttb.first()
        val bottomLeft = ltrttb.last()
        val topRight = rtlttb.first()
        val bottomRight = rtlttb.last()
        return listOf(topLeft, topRight, bottomRight, bottomLeft)
    }
}

data class Line(val first: Point, val second: Point) {
    fun angle(): Double = Math.abs(Math.atan2(first.y - second.y, first.x - second.x) *  180 / Math.PI)
}

operator fun Point.plus(other: Point): Point {
    return Point(this.x + other.x, this.y + other.y)
}

operator fun Point.times(other: Double): Point {
    return Point(this.x * other, this.y * other)
}

fun getQuads(contours: ArrayList<MatOfPoint>, minArea: Double): List<Contour> {
    return contours
            // Get approximated polygons
            .mapIndexed { index, matOfPoint ->
                // Convert to floats
                val contour2f = MatOfPoint2f()
                contours.get(index).convertTo(contour2f, CvType.CV_32F)

                // Approximate polygon
                val epsilon = Imgproc.arcLength(contour2f, true) * 0.1
                val poly2f = MatOfPoint2f()
                Imgproc.approxPolyDP(contour2f, poly2f, epsilon, true)

                // Convert to ints
                val poly = MatOfPoint()
                poly2f.convertTo(poly, CvType.CV_32S)

                // Return
                Contour(index, poly)
            }
            // Keep only 4-sided polys
            .filter { (_, matOfPoint) -> matOfPoint.total() == 4L }
            // Draw all polys in blue
            .map {
                contours.set(it.index, it.matOfPoint)
                it
            }
            .filter { it.area() >= minArea }
}