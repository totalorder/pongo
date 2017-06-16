package pongo
import org.opencv.core.*
import java.io.*
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc
import org.opencv.core.Mat
import org.opencv.core.Scalar


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

        val lines = findLines(quads)

        val straightLines = straightenLines(lines)

        straightLines.map {
            val color = Scalar(Math.random() * 255, Math.random() * 255, Math.random() * 255.0)
            drawLine(colored, it, color, 2)
        }

        return colored
    }


}