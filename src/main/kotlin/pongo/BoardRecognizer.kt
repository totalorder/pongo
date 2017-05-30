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

data class Contour(val index: Int, val matOfPoint: MatOfPoint) {
    fun area(): Double  = Imgproc.contourArea(matOfPoint)
}

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
        // Clone
        val gray = original.clone()

        // Grayscale
        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_RGB2GRAY)
        gray.convertTo(gray, CvType.CV_8U)

        // Equalize histogram
        Imgproc.equalizeHist(gray, gray)

        // Contrast
        contrast(gray, 0.1)

        // Blur
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.5)

        // Threshhold
        Imgproc.adaptiveThreshold(
                gray, gray, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 9, 4.0)

        // Dilation
        val dilationSize = 3.0
        val dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(dilationSize, dilationSize))
        Imgproc.dilate(gray, gray, dilateKernel)

        // Contours
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)

        // Convert to color for drawing colored polygons
        Imgproc.cvtColor(gray, gray, Imgproc.COLOR_GRAY2RGB)

        // Get the biggest two 4-sided polygons
        val biggestContours = contours
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
                    Imgproc.drawContours(gray, contours, it.index, Scalar(255.0, 0.0, 0.0), 1)
                    it
                }
                .sortedBy { it.area() }
                .takeLast(2)

        // Pick the grid out of two polys that might be either the board outline, or the grid outline
        // Get the smaller of the largest two polys if it's at least 90% of the area of the largest one,
        // otherwise get the largest poly
        val gridPoly = if (biggestContours.size == 2 && biggestContours[0].area() / biggestContours[1].area() > 0.9)
            biggestContours[0] else biggestContours[1]

        // Draw the outline of the grid
        Imgproc.drawContours(gray, contours, gridPoly.index, Scalar(0.0, 0.0, 255.0), 2)

        return gray
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


    /**
     * Return a function that returns a mat where one half is the original, and the other half the supplied mat
     */
    private fun compare(original: Mat): (mat: Mat) -> Mat {
        val clone = original.clone()
        return { sample: Mat ->
            sample.submat(0, clone.height(), 0, clone.width() / 2)
                    .copyTo(clone.submat(0, clone.height(), 0, clone.width() / 2))
            clone
        }
    }
}