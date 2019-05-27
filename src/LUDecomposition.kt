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
operator fun Matrix.set(i: Int, j: Int, value: Number) {
    this[i][j] = value.toFloat()
}

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
fun Matrix.LUDecomposition(): Pair<Matrix, Matrix> {
    val A = this
    val L = Array(size) { Array(size) { 0f } } // заполняем нулями
    val U = Array(size) { Array(size) { 0f } } // заполняем нулями

    for (j in 0 until size) {
        for (i in 0..j) {
            for (i_2 in 0..j) {
                U[i_2, j] = A[i_2, j]
                for (j_2 in 0 until i_2) {
                    U[i_2, j] -= L[i_2, j_2] * U[j_2, j]
                }
            }
            for (i_2 in j until size) {
                L[i_2, j] = A[i_2, j]
                for (j_2 in 0 until j) {
                    L[i_2, j] -= L[i_2, j_2] * U[j_2, j]
                }
                L[i_2, j] /= U[i, j]
            }
        }

    }
    return Pair(L, U)
}

fun main() {
    // исходная матрица
    val A = matrix(
        4,    8,    12,   -4,    2,
        8,    20,   12,   -6,    0,
       -8,   -8,   -39,    18,  -21,
        8,    12,   54,    3,   -9,
       -4,    0,   -63,   -9,    19
    )
    println(A)
    val (L, U) = A.LUDecomposition()
    print("L ")
    println(L)
    print("U ")
    println(U)
}
