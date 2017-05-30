package pongo
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


class BoardRecognizer {
    companion object {
        init {
            nu.pattern.OpenCV.loadShared()
        }
    }

    fun recognize(file: File) {
        val original = Highgui.imread(file.absolutePath, Highgui.CV_LOAD_IMAGE_COLOR)

        val output = File("${System.getProperty("java.io.tmpdir")}/test/recognize.jpg")

        File(output.parent).mkdirs()
        println(output.absolutePath)

        
        val processed = process(original)

        Highgui.imwrite(output.absolutePath, processed)
        Runtime.getRuntime().exec("eog ${output.absolutePath}")
    }

    private fun process(original: Mat): Mat {
        var gray = original.clone()

        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGB2GRAY)
        gray.convertTo(gray, CvType.CV_8U)
        println("channels: ${gray.channels()}")
        val grayComparer = compare(gray)
        Imgproc.equalizeHist(gray, gray)

        contrast(gray, 0.1)

        val contrastComparer = compare(gray)

        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.5)

//        return contrastComparer(gray)
        val blurComparer = compare(gray)
        Imgproc.adaptiveThreshold(
                gray, gray, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 9, 4.0)
//        Imgproc.threshold(gray, gray, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)

        val threshholdComparer = compare(gray)
//        return blurComparer(gray)

        val dilationSize = 3.0
        val dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(dilationSize, dilationSize))

        Imgproc.dilate(gray, gray, dilateKernel)
//        return threshholdComparer(gray)

//        Core.bitwise_not(gray, gray)

        val dilateComparer = compare(gray)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()

        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
//        return comparer(gray)
//        return dilateComparer(gray)

        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_GRAY2RGB)
        val contourComparer = compare(gray)
        val averageArea = contours.map { Imgproc.contourArea(it) }.average()

        data class Contour(val index: Int, val matOfPoint: MatOfPoint) {
            fun area(): Double {
                return Imgproc.contourArea(matOfPoint)
            }
        }

        val biggestContours = contours.mapIndexed { index, matOfPoint ->

            val contour2f = MatOfPoint2f()
            contours.get(index).convertTo(contour2f, CvType.CV_32F)

            val poly2f = MatOfPoint2f()
            val epsilon = Imgproc.arcLength(contour2f, true) * 0.1

            Imgproc.approxPolyDP(contour2f, poly2f,
                    epsilon, true)

            val poly = MatOfPoint()
            poly2f.convertTo(poly, CvType.CV_32S)
            Contour(index, poly)
        }
                .filter { (_, matOfPoint) -> matOfPoint.total() == 4L }
                .map {
                    contours.set(it.index, it.matOfPoint)
                    Imgproc.drawContours(gray, contours, it.index, Scalar(255.0, 0.0, 0.0), 1)
                    it
                }
                .sortedBy { it.area() }
                .takeLast(2)

        val biggest = if (biggestContours.size == 2
                && biggestContours[0].area() / biggestContours[1].area() > 0.9) {
            biggestContours[0]
        } else {
            biggestContours[1]
        }

        Imgproc.drawContours(gray, contours, biggest.index, Scalar(0.0, 0.0, 255.0), 2)

        return gray
    }

    private fun contrast(mat: Mat, strength: Double) {
        fun sigmoid(x: Int): Int {
            return (256.0 / (1.0 + Math.pow(Math.E, (-1.0 * strength) * (x - 128)))).toInt()
        }

        fun t(x: Int): Int {
            return (x + 256) % 256
        }
        fun ti(x: Int): Int {
            return (x - 128) % 256
        }

        val buff = ByteArray(mat.total().toInt())
        mat.get(0, 0, buff)

        buff.forEachIndexed { index, byte -> buff.set(index, (sigmoid(t(buff.get(index).toInt()))).toByte()) }

        for (x in 0..255) {
            val sig = sigmoid(x)
            buff.set(mat.cols() * sig + x, sig.toByte())
        }

        mat.put(0, 0, buff)
    }


    private fun compare(original: Mat): (mat: Mat) -> Mat {
        val clone = original.clone()
        return { sample: Mat ->
            sample.submat(0, clone.height(), 0, clone.width() / 2)
                    .copyTo(clone.submat(0, clone.height(), 0, clone.width() / 2))
            clone
        }
    }
}