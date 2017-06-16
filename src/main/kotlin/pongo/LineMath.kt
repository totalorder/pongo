package pongo

import org.apache.commons.math3.stat.regression.SimpleRegression
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar

fun linearRegression(line: List<Point>): Point {
    val sr = SimpleRegression()
    line.forEach { sr.addData(it.x, it.y) }
    return Point(1.0, sr.slope)
}

fun perpendicularLine(line: Point): Point {
    val slope = -1 / (line.y / line.x)
    return if(slope.isFinite()) Point(1.0, slope) else Point(0.0, 1.0)
}

fun drawLine(mat: Mat, line: List<Point>, color: Scalar, width: Int) {
    line.reduce({ first, second ->
        Core.line(mat, first, second, color, width)
        second
    })
}

fun intersection(first: Line, second: Line): Point? {
    val d = (first.first.x - first.second.x) * (second.first.y - second.second.y) -
            (first.first.y - first.second.y) * (second.first.x - second.second.x)
    if (d == 0.0) {
        return null
    }

    val xi = ((second.first.x - second.second.x) *
            (first.first.x * first.second.y - first.first.y * first.second.x) -
            (first.first.x - first.second.x) *
                    (second.first.x * second.second.y - second.first.y * second.second.x)) / d
    val yi = ((second.first.y - second.second.y) *
            (first.first.x * first.second.y - first.first.y * first.second.x) -
            (first.first.y - first.second.y) *
                    (second.first.x * second.second.y - second.first.y * second.second.x)) / d

    return Point(xi, yi)
}

fun distance(first: Point, second: Point): Double {
    return Math.hypot(first.x - second.x, first.y - second.y)
}