package pongo

import org.junit.Before
import org.junit.Test
import java.io.File

class BoardRecognizerTest {
    private lateinit var recognizer: BoardRecognizer

    @Before
    fun setUp() {
        recognizer = BoardRecognizer()
    }

    @Test
    fun recognizeBoard1() {
        recognizer.recognize(getFile("board1_crop.jpg"))
    }

    @Test
    fun recognizeBoard2() {
        recognizer.recognize(getFile("board2_crop.jpg"))
    }

    @Test
    fun recognizeBoard3() {
        recognizer.recognize(getFile("board3_crop.jpg"))
    }

    @Test
    fun recognizeBoard4() {
        recognizer.recognize(getFile("board4_crop.jpg"))
    }

    @Test
    fun testAll() {
        recognizeBoard1()
        recognizeBoard2()
        recognizeBoard4()
    }

    private fun getFile(name: String): File {
        return File(BoardRecognizerTest::class.java.classLoader.getResource(name).toURI())
    }

}