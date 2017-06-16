package pongo
import org.apache.commons.math3.stat.regression.SimpleRegression
import org.opencv.core.*
import java.io.*
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc
import java.util.ArrayList
import org.opencv.core.MatOfPoint
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.MatOfPoint2f
import org.opencv.core.CvType
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import java.util.Arrays
import org.opencv.core.Core


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

class BoardRecognizer {
    companion object {
        init {
            nu.pattern.OpenCV.loadShared()
        }
    }

    fun recognize(file: File) {
        val original = Highgui.imread(file.absolutePath, Highgui.CV_LOAD_IMAGE_COLOR)

        val output = File("${System.getProperty("java.io.tmpdir")}/test/${file.name}.jpg")

        File(output.parent).mkdirs()
        println(output.absolutePath)

        val start = System.currentTimeMillis()
        val processed = process(original)
        println("Processing time: ${System.currentTimeMillis() - start} ms")

        Highgui.imwrite(output.absolutePath, processed)
        Runtime.getRuntime().exec("eog ${output.absolutePath}")
    }

    private fun automat(block: (out: Mat) -> Unit): Mat {
        val out = Mat()
        block(out)
        return out
    }

    private fun grayscale(mat: Mat): Mat {
        val out = Mat()
        Imgproc.cvtColor(mat, out, Imgproc.COLOR_RGB2GRAY)
        out.convertTo(out, CvType.CV_8U)
        return out
    }

    private fun process(original: Mat): Mat {
        // Clone
        val clone = original.clone()

        val gray = grayscale(clone)

        val resized = resize(gray, 1024.0)

        val middleFiltered = filterMiddleColor(resized)

        val equalized = automat { Imgproc.equalizeHist(middleFiltered, it) }

        val blurred = automat { Imgproc.GaussianBlur(equalized, it, Size(5.0, 5.0), 0.5) }

        val thresholded = automat { Imgproc.adaptiveThreshold(
                blurred, it, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 65, -10.0) }

        val dilated = dilate(thresholded, 2.0)

        val contours = findContours(dilated)

        val colored = automat { Imgproc.cvtColor(dilated, it, Imgproc.COLOR_GRAY2RGB) }

        val quads = getQuads(contours, minArea = colored.size().area() / 1500.0)

        val lines = getLines(quads)

        val straightLines = getStraightLines(lines)

        straightLines.map {
            val color = Scalar(Math.random() * 255, Math.random() * 255, Math.random() * 255.0)
            drawLine(colored, it, color, 2)
        }

        return colored
    }

    private fun getStraightLines(lines: Map<Int, List<List<Point>>>): List<List<Point>> {
        return lines.values.flatMap { lines ->
            lines.map { line ->
                getStraightLine(line)
            }
        }
    }

    private fun getStraightLine(line: List<Point>): List<Point> {
        val sortedLine = line
                .sortedBy { distance(it, line.first()) }

        val regression = linearRegression(sortedLine)
        val firstToLast = linearRegression(listOf(sortedLine.first(), sortedLine.last()))

        val perpen = perpendicularLine(firstToLast)

        val lineCenter = Point(sortedLine.map { it.x }.average(), sortedLine.map { it.y }.average())
        val firstIntersection = intersection(Line(lineCenter, lineCenter + regression), Line(sortedLine.first(), sortedLine.first() + perpen))!!
        val lastIntersection = intersection(Line(lineCenter, lineCenter + regression), Line(sortedLine.last(), sortedLine.last() + perpen))!!
        return listOf(firstIntersection, lastIntersection)
    }

    private fun getLines(quads: List<Contour>): Map<Int, List<List<Point>>> {
        val averageArea = quads.map { it.area() }.average()
        val averageSideLength = Math.sqrt(averageArea)

        val rects = getRects(quads, averageArea, diffFactor = 0.4)
        val polysByPoint: Map<Int, List<Pair<Point, Rect>>> = getRectsBySide(rects)

        val lines = (0 until 4)
                .fold(mapOf<Int, List<List<Point>>>(), { acc: Map<Int, List<List<Point>>>, side: Int ->
                    // Map from side-id to list of Pair<Point, Rect>
                    val linesForSide = getLinesForSide(side, rects, polysByPoint, averageSideLength)
                    acc + Pair(side, linesForSide)
        })
        return lines
    }

    private fun getLinesForSide(side: Int,
                                rects: List<Rect>,
                                polysByPoint: Map<Int, List<Pair<Point, Rect>>>,
                                averageSideLength: Double): List<List<Point>> {
        val corner = getCorrespondingCorner(side)

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
        val line = lines[pointLines[point]]!!
        val otherPoint = findClosestPoint(polysByPoint, corner, point, averageSideLength)

        return if (otherPoint != null) {
            connectLines(lines, pointLines, otherPoint, point, line)
        } else {
            Pair(lines, pointLines)
        }
    }

    private fun connectLines(lines: Map<Point, List<Point>>, pointLines: Map<Point, Point>, otherPoint: Point, point: Point, line: List<Point>): Pair<Map<Point, List<Point>>, Map<Point, Point>> {
        val otherLine = lines[pointLines[otherPoint]]!!
        val newLines = lines - pointLines[otherPoint]!! + Pair(pointLines[point]!!, line + otherLine)
        val newPointLines = pointLines + otherLine.map { Pair(it, pointLines[point]!!) }
        return Pair(newLines, newPointLines)
    }

    private fun findClosestPoint(polysByPoint: Map<Int, List<Pair<Point, Rect>>>, corner: Int, point: Point, averageSideLength: Double): Point? {
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

    private fun getRectsBySide(perfectRects: List<Rect>): Map<Int, List<Pair<Point, Rect>>> {
        return perfectRects.fold(mapOf<Int, List<Pair<Point, Rect>>>(), { acc, poly ->
            (0 until 4).fold(acc, { subacc, i: Int ->
                subacc + Pair(i, acc.getOrDefault(i, listOf()) + Pair(poly.points[i], poly))
            })
        })
    }

    private fun getQuads(contours: ArrayList<MatOfPoint>, minArea: Double): List<Contour> {
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

    private fun findContours(mat: Mat): ArrayList<MatOfPoint> {
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(mat.clone(), contours, Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
        return contours
    }

    private fun dilate(mat: Mat, dilationSize: Double): Mat {
        val out = Mat()
        val dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(dilationSize, dilationSize))
        Imgproc.dilate(mat, out, dilateKernel)
        return out
    }

    private fun resize(mat: Mat, maxSize: Double): Mat {
        val out = Mat()
        if (mat.width() > maxSize || mat.height() > maxSize) {
            Imgproc.resize(mat, out, Size(maxSize, maxSize), 0.0, 0.0, Imgproc.INTER_LINEAR)
            return out
        }
        return mat
    }

    fun filterMiddleColor(mat: Mat, numberOfBins: Int = 60): Mat {
        val out = mat.clone()
        // Filter away the colors lighter than the middle color
        data class Bin(val index: Int, val value: Double)

        // Calculate histogram
        val hist = Mat()
        Imgproc.calcHist(Arrays.asList(mat), MatOfInt(0),
                Mat(), hist, MatOfInt(numberOfBins), MatOfFloat(0.0f, 255.0f))
        val allBins = (0 until numberOfBins)
                .map {
                    Bin(it, hist.get(it, 0)[0])
                }
        val distanceThreshold = 0.5
        val spanThreshold = 3.5
        val selectedBins = allBins
                .filter { it.index > numberOfBins / spanThreshold && it.index < numberOfBins - numberOfBins / spanThreshold }
        val maxSelectedBin = selectedBins.reduce({left, right -> if (left.value > right.value) left else right})
        val maxBin = allBins.map { it.value }.max()!!
        for (binIndex in (selectedBins[0].index..maxSelectedBin.index).reversed()) {
            val bin = allBins[binIndex]

            if (bin.value < maxSelectedBin.value * distanceThreshold &&
                    binIndex + 1 < allBins.size &&
                    bin.value > allBins[binIndex + 1].value) {
                break
            }

            applyOnPixels(out, { value -> if (value < 255 / numberOfBins * bin.index) value else 255 })

            if (false) {
                // Draw histogram
                val barHeight = out.height() * (bin.value / maxBin)
                val barX = (out.width() / (numberOfBins + 1)) * bin.index
                val width = out.width() / ((numberOfBins + 1) * 2).toDouble()

                Core.rectangle(out,
                        Point(barX + width, out.height() - barHeight),
                        Point(barX + width * 2, out.height().toDouble()),
                        Scalar(180.0 + (bin.index % 2 * 60), 0.0, 0.0),
                        -1
                )
            }


        }
        return out
    }


    /**
     * Get the bounding right-angled rectangle (starting at origin)
     * of a non-right-angled rectangle
     */
    private fun boundingRect(rect: List<Point>): List<Point> {
        val maxWidth = Math.max(distance(rect[0], rect[1]), distance(rect[2], rect[3]))
        val maxHeight = Math.max(distance(rect[0], rect[2]), distance(rect[1], rect[3]))

        return listOf(
                Point(0.0, 0.0),
                Point(maxWidth, 0.0),
                Point(0.0, maxHeight),
                Point(maxWidth, maxHeight))
    }

    /**
     * Get the bounding square of a right-angled rectangle starting at origin
     */
    private fun maxSquare(rect: List<Point>): List<Point> {
        val max = Math.max(rect[3].x, rect[3].y)
        return listOf(
                Point(0.0, 0.0),
                Point(max, 0.0),
                Point(0.0, max),
                Point(max, max))
    }

    /**
     * Get the distance between two points
     */
    private fun distance(first: Point, second: Point): Double {
        return Math.hypot(first.x - second.x, first.y - second.y)
    }

    /**
     * Sort the points of a rectangle in the order of:
     * top left -> top right -> bottom left -> bottom -> right
     */
    private fun sortedRectangle(rectPoints: MutableList<Point>): List<Point> {
        val summedPoints = rectPoints.sortedBy { point -> point.x + point.y }
        val topLeft = summedPoints.first()
        val bottomRight = summedPoints.last()
        val diffedPoints = rectPoints.sortedBy { point -> point.x - point.y }
        val topRight = diffedPoints.last()
        val bottomLeft = diffedPoints.first()
        return listOf(topLeft, topRight, bottomLeft, bottomRight)
    }

    /**
     * Increase the contrast of mat. Assumes the Mat to be in CV_8S
     */
    private fun contrast(mat: Mat, strength: Double) {
        /**
         * Return the y-value on position x on the Logistic function sigmoid in the 0..255-range
         */
        fun sigmoid(x: Int): Int {
            return (256.0 / (1.0 + Math.pow(Math.E, (-1.0 * strength) * (x - 128)))).toInt()
        }

        /**
         * Convert the signed 8-bit int x to unsigned
         */
        fun cv8STo8U(x: Int): Int {
            return (x + 256) % 256
        }

        /**
         * Convert the unsigned 8-bit int x to signed
         * TODO: Might not work :)
         */
        fun cv8UTo8S(x: Int): Int {
            return (x - 128) % 256
        }

        // Put all bytes of mat into buff
        val buff = ByteArray(mat.total().toInt())
        mat.get(0, 0, buff)

        // Apply the sigmoid function to each byte in buff
        buff.forEachIndexed { index, byte -> buff.set(index, (sigmoid(cv8STo8U(buff.get(index).toInt()))).toByte()) }

        // Put all bytes back into mat
        mat.put(0, 0, buff)
    }

    private fun applyOnPixels(mat: Mat, block: (value: Int) -> Int) {
        val buff = ByteArray(mat.total().toInt() * mat.channels())
        mat.get(0, 0, buff)
        buff.forEachIndexed { index, byte -> buff[index] = block(byte.toInt() and 0xFF).toByte() }
        mat.put(0, 0, buff)
    }

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
}