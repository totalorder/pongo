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
import org.opencv.utils.Converters
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

    private fun process(original: Mat): Mat {
        if (false) {
            // Uncrop square
            val bound = Math.max(original.width(), original.height()).toDouble()
            val cropLeft = (bound - original.width()).toInt() / 2
            val cropRight = (bound - original.width()).toInt() - cropLeft
            val cropTop = (bound - original.height()).toInt() / 2
            val cropBottom = (bound - original.height()).toInt() - cropTop
            Imgproc.copyMakeBorder(original, original, cropTop, cropBottom, cropLeft, cropRight, Imgproc.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
        }

        // Clone
        val gray = original.clone()

        // Grayscale
        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGB2GRAY)
        gray.convertTo(gray, CvType.CV_8U)

        if (true) {
            // Resize
            val maxSize = 1024.0
            if (gray.width() > maxSize || gray.height() > maxSize) {

                Imgproc.resize(gray, gray, Size(maxSize, maxSize), 0.0, 0.0, Imgproc.INTER_LINEAR)
            }
        }

        // Filter away the colors lighter than the middle color
        filterMiddleColor(gray)

        // Equalize histogram
        Imgproc.equalizeHist(gray, gray)
        val preprocessed = gray.clone()

        // Blur
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.5)

        if (false) {
            // Apply contrast
            applyOnPixels(gray, { value -> Math.min(Math.round(value * value * 0.0195).toInt(), 255) })
        }

        // Threshold
        Imgproc.adaptiveThreshold(
                gray, gray, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 65, -10.0)


        // Dilation
        val dilationSize = 2.0
        val dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(dilationSize, dilationSize))
        Imgproc.dilate(gray, gray, dilateKernel)

        if (false) {
            // Flood fill anything that touches the border
            (0 until Math.max(gray.width(), gray.height()))
                    .map {
                        val top = Point(it.toDouble(), 0.0)
                        val left = Point(0.0, it.toDouble())
                        val right = Point(gray.width().toDouble() - 1, it.toDouble())
                        val bottom = Point(it.toDouble(), gray.height().toDouble() - 1)

                        listOf(top, left, right, bottom).map {
                            if (it.x < gray.width() && it.y < gray.height()) {
                                val floodMask = Mat()
                                Imgproc.floodFill(gray, floodMask, it, Scalar(0.0, 0.0, 0.0))
                            }
                        }
                    }
        }

        // Draw corner zone
        val perspectiveRatio = 2.7
        val heightRatio = 1.2
        val inset = 0.06
        val cornerZonePoly = listOf(MatOfPoint(
                Point(gray.width() * inset * perspectiveRatio, gray.height() * inset * perspectiveRatio * heightRatio),
                Point(gray.width() - gray.width() * inset * perspectiveRatio, gray.height() * inset * perspectiveRatio * heightRatio),
                Point(gray.width() - gray.width() * inset, gray.height() - gray.height() * inset * perspectiveRatio * heightRatio),
                Point(gray.width() * inset, gray.height() - gray.height() * inset * perspectiveRatio * heightRatio)))
//        Core.fillPoly(gray, cornerZonePoly, Scalar(0.0, 0.0, 0.0))

        // Contours
        val contouredMat = gray.clone()

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(contouredMat, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

        // Convert to color for drawing colored polygons
        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_GRAY2RGB)

        // Get the biggest two 4-sided polygons
        val polygons = contours
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
                    Imgproc.drawContours(gray, contours, it.index, Scalar(0.0, 50.0, 255.0), 1)
                    it
                }
                .sortedBy { it.area() }

        val bigPolygons = polygons.filter {
            it.area() > gray.size().area() / 1500.0
        }.map {
            Imgproc.drawContours(gray, contours, it.index, Scalar(255.0, 50.0, 50.0), 1)
            it
        }

        val averagePolySize = bigPolygons.map { it.area() }.average()
        val perfectPolygons = bigPolygons
                .filter { Math.abs(it.area() - averagePolySize) < averagePolySize * 0.4 }
                .map {
                    Imgproc.drawContours(gray, contours, it.index, Scalar(50.0, 255.0, 50.0), 1)
                    it
                }.map {
                    val rect = Rect(it.matOfPoint.toArray().toList())
                    if (rect.points.distinct().count() != 4) {
                        drawLine(gray, it.matOfPoint.toArray().toList(), Scalar(50.0, 0.0, 255.0), 3)
                    }
                    rect
                }
//                .sortedBy { it.points[0].x + it.points[0].y }
//                .take(5)

        val averageSideLength = Math.sqrt(averagePolySize)

        val polysByPoint = perfectPolygons.fold(mapOf<Int, List<Pair<Point, Rect>>>(), { acc, poly ->
            (0 until 4).fold(acc, { subacc, i: Int ->
                subacc + Pair(i, acc.getOrDefault(i, listOf()) + Pair(poly.points[i], poly))
            })
        })

        val lines = (0 until 4).fold(mapOf<Int,List<List<Point>>>(), { outerAcc: Map<Int,List<List<Point>>>, side: Int ->
            val direction = when(side) {
                0 -> 3
                1 -> 2
                2 -> 0
                3 -> 3
                else -> throw IllegalArgumentException("Invalid value: " + side)
            }
            val sideLines = perfectPolygons.fold(mapOf<Point, List<Point>>(), { acc, poly ->
                val line = when(side) {
                    0 -> listOf(poly.points[3], poly.points[0])
                    1 -> listOf(poly.points[2], poly.points[1])
                    2 -> listOf(poly.points[0], poly.points[1])
                    3 -> listOf(poly.points[3], poly.points[2])
                    else -> throw IllegalArgumentException("Invalid value: " + side)
                }

                acc + Pair(line.last(), line)
            })

            val pointLines = sideLines.entries.fold(mapOf<Point, Point>(), { acc, line ->
                acc + line.value.map { Pair(it, line.key) }
            })

            val mergedLines = sideLines.keys.fold(Pair(sideLines, pointLines), { (lines, pointLines), point ->
                val line = lines[pointLines[point]]!!
                val otherPoint = polysByPoint[direction]!!
                        .map { Pair(distance(it.first, point), it.first) }
                        .filter { it.first < averageSideLength * 0.4 }
                        .sortedBy { it.first }
                        .map { it.second }
                        .takeLast(1)
                        .getOrNull(0)

                if (otherPoint != null) {
                    val otherLine = lines[pointLines[otherPoint]]!!
                    val newLines = lines - pointLines[otherPoint]!! + Pair(pointLines[point]!!, line + otherLine)
                    val newPointLines = pointLines + otherLine.map { Pair(it, pointLines[point]!!) }

                    // Detect duplicates
                    lines.entries
                            .map { entry ->
                                entry.value.map {
                                    Pair(entry.key, it) } }
                            .flatMap { it }
                            .groupBy { it.second }.entries
                            .filter { it.value.size > 1 }
                            .map {
                                println("Double! ${it}")
                            }

                    Pair(newLines, newPointLines)
                } else {
                    Pair(lines, pointLines)
                }
            })

            outerAcc + Pair(side, mergedLines.first.values.filter { it.size > 3 }.toList())
        })

        lines.values.flatMap { lines ->
            lines.map { line ->
                val color = Scalar(Math.random() * 255, Math.random() * 255, Math.random() * 255.0)
                val sortedLine = line
                        .sortedBy { distance(it, line.first()) }

                val regression = linearRegression(sortedLine)
                val firstToLast = linearRegression(listOf(sortedLine.first(), sortedLine.last()))

                val perpen = perpendicularLine(firstToLast)

                val lineCenter = Point(sortedLine.map { it.x }.average(), sortedLine.map { it.y }.average())
                val firstIntersection = intersection(Line(lineCenter, lineCenter + regression), Line(sortedLine.first(), sortedLine.first() + perpen))!!
                val lastIntersection = intersection(Line(lineCenter, lineCenter + regression), Line(sortedLine.last(), sortedLine.last() + perpen))!!
                drawLine(gray, listOf(firstIntersection, lastIntersection), color, 2)

            }
        }

        return gray

        val biggestContours = polygons


        // Pick the grid out of two polys that might be either the board outline, or the grid outline
        // Get the smaller of the largest two polys if it's at least 90% of the area of the largest one,
        // otherwise get the largest poly
        val gridPoly = if (biggestContours.size == 2 && biggestContours[0].area() / biggestContours[1].area() > 0.9)
            biggestContours[0] else biggestContours[1]
        // Draw the outline of the grid
        Imgproc.drawContours(gray, contours, gridPoly.index, Scalar(0.0, 0.0, 255.0), 2)

        return gray

        if (gridPoly.area() / gray.size().area() < 0.3) {
            println("Did not find the board!")
            return gray
//            return originalComparer(gray, true)
        }

//        return gray

        // Unwarp perspective
        // Get mats representing the current perspective rectangle, and the goal square
        val originalRect = sortedRectangle(gridPoly.matOfPoint.toList())
        val originalMat = Converters.vector_Point2f_to_Mat(originalRect)
        val originalDstSquare = maxSquare(boundingRect(originalRect))
        val originalDstMat = Converters.vector_Point2f_to_Mat(originalDstSquare)


        // Warp between src perspective rectangle and dst square
        val perspectiveTransform = Imgproc.getPerspectiveTransform(originalMat, originalDstMat)
        Imgproc.warpPerspective(preprocessed, preprocessed, perspectiveTransform, Size(originalDstSquare[3].x, originalDstSquare[3].x))

        // Draw the spaces which might include stones
        val gridSize = preprocessed.width() / 18.0
        // Convert to color for drawing colored polygons
        Imgproc.cvtColor(preprocessed, preprocessed, Imgproc.COLOR_GRAY2RGB)
        for (x in 0..19) {
            for (y in  0..19) {
                val center = Point(x * gridSize, y * gridSize)
                val topLeft = Point(x * gridSize - gridSize / 2, y * gridSize - gridSize / 2)

                if (x in 0..18 && y in 0..18) {
                    val interestRadius = gridSize * 0.9 / 2

                    // Fill area around interest circle with gray
                    val fullRectangle = Rect(
                            Point(topLeft.x, topLeft.y),
                            Point(topLeft.x + gridSize, topLeft.y + gridSize))
                    val fullMask = Mat(preprocessed.size(), CvType.CV_8U, Scalar(255.0, 255.0, 255.0))
                    Core.rectangle(fullMask, fullRectangle.tl(), fullRectangle.br(), Scalar(0.0, 0.0, 0.0), -1)
                    Core.circle(fullMask, center, (interestRadius).toInt(), Scalar(255.0, 255.0, 255.0), -1)
                    Core.bitwise_not(fullMask, fullMask)
                    val grayMat = Mat(preprocessed.size(), CvType.CV_8U, Scalar(127.0, 127.0, 127.0))
                    Imgproc.cvtColor(grayMat, grayMat, Imgproc.COLOR_GRAY2RGB) // TODO: Might exist a better way
                    grayMat.copyTo(preprocessed, fullMask)

                    // Create a material over the interest area
                    val interestRectangle = Rect(
                            Point(Math.round(Math.max(center.x - interestRadius, 0.0)).toDouble(),
                                    Math.round(Math.max(center.y - interestRadius, 0.0)).toDouble()),
                            Point(Math.round(Math.min(center.x + interestRadius, preprocessed.width().toDouble())).toDouble(),
                                    Math.round(Math.min(center.y + interestRadius, preprocessed.height().toDouble())).toDouble()))
                    val interestMat = Mat(preprocessed, interestRectangle)

                    // Calculate histogram
                    val hist = Mat()
                    val fullNumberOfBins = 23
                    Imgproc.calcHist(Arrays.asList(interestMat), MatOfInt(0),
                            Mat(), hist, MatOfInt(fullNumberOfBins), MatOfFloat(0.0f, 255.0f))
                    val histogramBins = (0 until fullNumberOfBins)
                            .filterNot { it == fullNumberOfBins / 2 } // Remove center bin because of grayness
                            .map {
                        hist.get(it, 0)[0]
                    }
                    val numberOfBins = histogramBins.size

                    // Draw histogram
                    val maxBin = histogramBins.max()!!
                    histogramBins.mapIndexed { index, value ->
                        val barHeight = gridSize * (value / maxBin)
                        val barX = (gridSize / (numberOfBins + 1)) * index
                        val width = gridSize / ((numberOfBins + 1) * 2)

                        Core.rectangle(preprocessed,
                                Point(topLeft.x + barX + width, topLeft.y + gridSize - barHeight),
                                Point(topLeft.x + barX + width * 2, topLeft.y + gridSize),
                                Scalar(180.0 + (index % 2 * 60), 0.0, 0.0),
                                -1
                        )
                    }

                    // Draw guess
                    val sum = histogramBins.sum()

                    val threshold = 0.5
                    if (histogramBins.first() > histogramBins.last()) { // Black
                        if ((histogramBins[0] + histogramBins[1] + histogramBins[2] + histogramBins[3] + histogramBins[4]) / sum > threshold) {
                            Core.rectangle(preprocessed, Point(fullRectangle.x+1.0, fullRectangle.y+1.0), Point(fullRectangle.br().x-1, fullRectangle.br().y-1),
                                    Scalar(0.0, 0.0, 0.0)
                            )
                        }
                    } else { // White
                        if (histogramBins[numberOfBins-1] / sum > threshold) {
                            Core.rectangle(preprocessed, Point(fullRectangle.x+1.0, fullRectangle.y+1.0), Point(fullRectangle.br().x-1, fullRectangle.br().y-1),
                                    Scalar(255.0, 255.0, 255.0)
                            )
                        }
                    }
                }
            }
        }

        return preprocessed
    }

    fun filterMiddleColor(mat: Mat, numberOfBins: Int = 60) {
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

            applyOnPixels(mat, { value -> if (value < 255 / numberOfBins * bin.index) value else 255 })

            if (false) {
                // Draw histogram
                val barHeight = mat.height() * (bin.value / maxBin)
                val barX = (mat.width() / (numberOfBins + 1)) * bin.index
                val width = mat.width() / ((numberOfBins + 1) * 2).toDouble()

                Core.rectangle(mat,
                        Point(barX + width, mat.height() - barHeight),
                        Point(barX + width * 2, mat.height().toDouble()),
                        Scalar(180.0 + (bin.index % 2 * 60), 0.0, 0.0),
                        -1
                )
            }


        }
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


    /**
     * Return a function that returns a mat where one half is the original, and the other half the supplied mat
     */
    private fun compare(original: Mat): (mat: Mat, toColor: Boolean) -> Mat {
        val clone = original.clone()
        return { sample: Mat, toColor: Boolean ->
            if (toColor) {
                Imgproc.cvtColor(clone, clone, Imgproc.COLOR_GRAY2RGB)
            }
            sample.submat(0, clone.height(), 0, clone.width() / 2)
                    .copyTo(clone.submat(0, clone.height(), 0, clone.width() / 2))
            clone
        }
    }

    fun largestSecondRootSizeUnderRoof(size: Size, roof: Double): Size {
        val max = Math.max(size.height, size.width)
        val divisor = Math.pow(2.0, Math.ceil(Math.log(max / roof) / Math.log(2.0)))
        return Size(size.width / divisor, size.height / divisor)
    }

    fun linearRegression(line: List<Point>): Point {
        val sr = SimpleRegression()
        line.forEach { sr.addData(it.x, it.y) }
        return Point(1.0, sr.slope)

        val p1 = Point(line.map { it.x }.average(), line.map { it.y }.average())
        
//        var xp1.x = 0.0
//        var yp1.y = 0.0
//        var xp1.y = 0.0
        val x = line.map { (it.x - p1.x) * (it.x - p1.x) }.sum()
//        val y = line.map { (it.y - p1.y) * (it.y - p1.y) }.sum()
        val xy = line.map { (it.x - p1.x) * (it.y - p1.y) }.sum()
        val beta1 = xy / x
        val beta0 = p1.y - beta1 * p1.x
        
        return Point(beta0, beta1)
    }

    fun perpendicularLine(line: Point): Point {
        val slope = -1 / (line.y / line.x)
//        val angle = Math.atan2(line.y, line.x)
//        val distance = 20
//        val x = Math.sin(angle) * distance
//        val y = Math.cos(angle) * distance
        return Point(1.0, slope)
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