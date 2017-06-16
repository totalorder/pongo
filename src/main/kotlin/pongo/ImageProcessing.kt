package pongo

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.*

fun automat(block: (out: Mat) -> Unit): Mat {
    val out = Mat()
    block(out)
    return out
}

fun findContours(mat: Mat): ArrayList<MatOfPoint> {
    val contours = ArrayList<MatOfPoint>()
    Imgproc.findContours(mat.clone(), contours, Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
    return contours
}

fun dilate(mat: Mat, dilationSize: Double): Mat {
    val out = Mat()
    val dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(dilationSize, dilationSize))
    Imgproc.dilate(mat, out, dilateKernel)
    return out
}

fun resize(mat: Mat, maxSize: Double): Mat {
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
            .filter { it.index > numberOfBins / spanThreshold &&
                    it.index < numberOfBins - numberOfBins / spanThreshold }
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

fun grayscale(mat: Mat): Mat {
    val out = Mat()
    Imgproc.cvtColor(mat, out, Imgproc.COLOR_RGB2GRAY)
    out.convertTo(out, CvType.CV_8U)
    return out
}

private fun applyOnPixels(mat: Mat, block: (value: Int) -> Int) {
    val buff = ByteArray(mat.total().toInt() * mat.channels())
    mat.get(0, 0, buff)
    buff.forEachIndexed { index, byte -> buff[index] = block(byte.toInt() and 0xFF).toByte() }
    mat.put(0, 0, buff)
}