package pongo

import org.opencv.core.Point

fun straightenLines(lines: Map<Int, List<List<Point>>>): List<List<Point>> {
    return lines.values.flatMap { lines ->
        lines.map { line ->
            getStraightLine(line)
        }
    }
}

fun findLines(quads: List<Contour>): Map<Int, List<List<Point>>> {
    val averageArea = quads.map { it.area() }.average()
    val averageSideLength = Math.sqrt(averageArea)

    // Rects close to the average size
    val rects = getRects(quads, averageArea, diffFactor = 0.4)

    // Map from side-id to list of Pair<Point, Rect>
    val polysByPoint: Map<Int, List<Pair<Point, Rect>>> = getRectsBySide(rects)

    // Map of side-id to list of lines
    val lines: Map<Int, List<List<Point>>> = (0 until 4)
            .fold(mapOf<Int, List<List<Point>>>(), { acc: Map<Int, List<List<Point>>>, side: Int ->
                val linesForSide: List<List<Point>> = getLinesForSide(side, rects, polysByPoint, averageSideLength)
                acc + Pair(side, linesForSide)
            })
    return lines
}

private fun getStraightLine(line: List<Point>): List<Point> {
    val sortedLine = line
            .sortedBy { distance(it, line.first()) }

    val regression: Point = linearRegression(sortedLine)
    val firstToLast: Point = linearRegression(listOf(sortedLine.first(), sortedLine.last()))

    val perpenLine: Point = perpendicularLine(firstToLast)

    val lineCenter = Point(sortedLine.map { it.x }.average(), sortedLine.map { it.y }.average())

    val firstIntersection = intersection(
            Line(lineCenter, lineCenter + regression),
            Line(sortedLine.first(), sortedLine.first() + perpenLine))!!
    val lastIntersection = intersection(
            Line(lineCenter, lineCenter + regression),
            Line(sortedLine.last(), sortedLine.last() + perpenLine))!!
    return listOf(firstIntersection, lastIntersection)
}



private fun getLinesForSide(side: Int,
                            rects: List<Rect>,
                            polysByPoint: Map<Int, List<Pair<Point, Rect>>>,
                            averageSideLength: Double): List<List<Point>> {
    val corner: Int = getCorrespondingCorner(side)

    val sideLines: Map<Point, List<Point>> = getSideLines(rects, side)

    val pointLines: Map<Point, Point> = getPointLines(sideLines)

    val mergedLines: Pair<Map<Point, List<Point>>, Map<Point, Point>> = sideLines.keys.fold(
            Pair(sideLines, pointLines), { (lines, pointLines), point ->
        connectCloseLines(lines, pointLines, point, polysByPoint, corner, averageSideLength)
    })
    return mergedLines.first.values.filter { it.size > 3 }.toList()
}

private fun connectCloseLines(
        lines: Map<Point, List<Point>>,
        pointLines: Map<Point, Point>,
        point: Point, polysByPoint: Map<Int, List<Pair<Point, Rect>>>,
        corner: Int, averageSideLength: Double): Pair<Map<Point, List<Point>>, Map<Point, Point>> {
    val line: List<Point> = lines[pointLines[point]]!!
    val otherPoint: Point? = findClosestPoint(polysByPoint, corner, point, averageSideLength)

    return if (otherPoint != null) {
        connectLines(lines, pointLines, otherPoint, point, line)
    } else {
        Pair(lines, pointLines)
    }
}

private fun connectLines(lines: Map<Point, List<Point>>,
                         pointLines: Map<Point, Point>,
                         otherPoint: Point,
                         point: Point,
                         line: List<Point>): Pair<Map<Point, List<Point>>, Map<Point, Point>> {
    val otherLine: List<Point> = lines[pointLines[otherPoint]]!!
    val newLines: Map<Point, List<Point>> = lines - pointLines[otherPoint]!! +
            Pair(pointLines[point]!!, line + otherLine)
    val newPointLines: Map<Point, Point> = pointLines + otherLine.map { Pair(it, pointLines[point]!!) }
    return Pair(newLines, newPointLines)
}

private fun findClosestPoint(polysByPoint: Map<Int, List<Pair<Point, Rect>>>,
                             corner: Int,
                             point: Point,
                             averageSideLength: Double): Point? {
    return polysByPoint[corner]!!
            .map { Pair(distance(it.first, point), it.first) }
            .filter { it.first < averageSideLength * 0.4 }
            .sortedBy { it.first }
            .map { it.second }
            .takeLast(1)
            .getOrNull(0)
}

private fun getPointLines(sideLines: Map<Point, List<Point>>): Map<Point, Point> {
    return sideLines.entries.fold(mapOf<Point, Point>(), { acc, line ->
        acc + line.value.map { Pair(it, line.key) }
    })
}

private fun getSideLines(rects: List<Rect>, side: Int): Map<Point, List<Point>> {
    return rects.fold(mapOf<Point, List<Point>>(), { acc, poly ->
        val sideLine = getSideLine(side, poly)
        acc + Pair(sideLine.last(), sideLine)
    })
}

private fun getSideLine(side: Int, poly: Rect): List<Point> {
    return when (side) {
        0 -> listOf(poly.points[3], poly.points[0]) // (left side)   bottomLeft, topLeft
        1 -> listOf(poly.points[2], poly.points[1]) // (right side)  bottomRight, topRight
        2 -> listOf(poly.points[0], poly.points[1]) // (top side)    topLeft, topRight
        3 -> listOf(poly.points[3], poly.points[2]) // (bottom side) bottomLeft, bottomRight
        else -> throw IllegalArgumentException("Invalid value: " + side)
    }
}

private fun getCorrespondingCorner(side: Int): Int {
    return when (side) {
        0 -> 3 // (left side)   topLeft -> bottomLeft
        1 -> 2 // (right side)  topRight -> bottomRight
        2 -> 0 // (top side)    topRight -> topLeft
        3 -> 3 // (bottom side) bottomRight -> bottomLeft
        else -> throw IllegalArgumentException("Invalid value: " + side)
    }
}

private fun getRects(quads: List<Contour>, averageArea: Double, diffFactor: Double): List<Rect> {
    return quads
            .filter { Math.abs(it.area() - averageArea) < averageArea * diffFactor }
            .map {
                Rect(it.matOfPoint.toArray().toList())
            }
}

private fun getRectsBySide(rects: List<Rect>): Map<Int, List<Pair<Point, Rect>>> {
    return rects.fold(mapOf<Int, List<Pair<Point, Rect>>>(), { acc, poly ->
        (0 until 4).fold(acc, { subacc, i: Int ->
            subacc + Pair(i, acc.getOrDefault(i, listOf()) + Pair(poly.points[i], poly))
        })
    })
}