import kotlin.math.sqrt

/**
 * Матрица это массив из массивов вещественных чисел
 */
typealias Matrix = Array<Array<Float>>

/**
 * Конструктор для упрощенного создания матрицы (только квадратной)
 */
fun matrix(vararg e: Number): Matrix {
    val size = sqrt(e.size.toDouble()).toInt()
    var i = 0
    return Array(size) { Array(size) {e[i++].toFloat()} }
}

/**
 * Добавляем возможность обращаться к матрицам при помощи двух индексов через запятую: А[i, j]
 */
operator fun Matrix.get(i: Int, j: Int) = this[i][j]

/**
 * Функция для преобразования матрицы в текст (чтоб можно было напечатать в консоли)
 */
fun Matrix.asString(): String {
    var s = "Matrix (${size}x$size):\n"
    for (i in 0 until this.size) {
        for (j in 0 until this.size) {
            val x = String.format("%1$10s", this[i, j])
            s += "$x "
        }
        s+= "\n"
    }
    return s
}

fun println(matrix: Matrix) = println(matrix.asString())

/**
 * LU-разложение матрицы
 */
//fun Matrix.LUDecompose(): Pair<Matrix, Matrix> {
//
//}

fun main() {
    val A = matrix(
        1.555,  2, 3,
        4,      5, 6,
        7,      8, 9
    )
    println(A)
}
