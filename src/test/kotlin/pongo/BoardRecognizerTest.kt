package pongo

import org.junit.Before
import org.junit.Test
import java.io.File

class BoardRecognizerTest {
    val fileName = "board.jpg"
    val input = File(BoardRecognizerTest::class.java.classLoader.getResource(fileName).toURI())!!

    private lateinit var recognizer: BoardRecognizer

    @Before
    fun setUp() {
        recognizer = BoardRecognizer()
    }

    @Test
    fun recognize() {

        recognizer.recognize(input)

    }

}