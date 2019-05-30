import kotlin.math.sqrt

/**
 * Матрица это массив из массивов вещественных чисел
 */
typealias Matrix = Array<Array<Float>>

/**
 * Конструктор для упрощенного создания матрицы (только квадратной)
 */
fun matrix(vararg e: Number): Matrix {
    val size = sqrt(e.size.toFloat()).toInt()
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
 * Умножение матриц
 */
fun matmul(A: Matrix, B: Matrix): Matrix {
    if (A.size != B.size) throw ArithmeticException("Matrix sizes does not equal")
    val C = Array(A.size) { Array(A.size) { 0f } }

    for (i in 0 until A.size) {
        for (j in 0 until A.size) {
            for (k in 0 until A.size) {
                C[i, j] += A[i, k] * B[k, j]
            }
        }
    }

    return C
}

/**
 * Транспонирование матриц
 */
fun Matrix.transpose(): Matrix {
    val m = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            m[i, j] = this[j, i]
        }
    }
    return m
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
fun Matrix.decomposeLU(): Pair<Matrix, Matrix> {
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

/**
 * Разложение Холецкого
 */
fun Matrix.cholesky(): Matrix {
    val A = this
    val L = Array(size) { Array(size) { 0f } } // заполняем нулями

    for (j in 0 until size) {
        L[j, j] = A[j, j]
        for (i in 0 until j) {
            L[j, j] -= L[j, i] * L[j, i]
        }
        L[j, j] = sqrt(L[j, j])

        for (i in (j+1) until size) {
            L[i, j] = A[i, j]
            for (j_2 in 0 until j) {
                L[i, j] -= L[i, j_2] * L[j, j_2]
            }
            L[i, j] /= L[j, j]
        }
    }

    return L
}

fun generateForCholesky(size: Int): Pair<Matrix, Matrix> {
    val variables = arrayOf(-5, -4, -3, -2, -1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5)
    val L = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0..i) {
            L[i, j] = variables.random()
        }
    }
    return Pair(L, matmul(L, L.transpose()))
}

fun generateForLU(size: Int): Triple<Matrix, Matrix, Matrix> {
    val variables = arrayOf(-5, -4, -3, -2, -1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5)
    val L = Array(size) { Array(size) { 0f } }
    val U = Array(size) { Array(size) { 0f } }

    for (i in 0 until size) {
        for (j in 0 until i) {
            L[i, j] = variables.random()
        }
        L[i, i] = 1
    }

    for (i in 0 until size) {
        for (j in i until size) {
            U[i, j] = variables.random()
        }
    }
    return Triple(matmul(L, U), L, U)
}


fun main() {
    repeat(3) {
        val (L, A) = generateForCholesky(4)
        println("\nРазложение Холецкого ${it+1}\n")

        print("Задача: А ")
        println(A)
        print("Ответ: L ")
        println(L)
    }
    repeat(3) {
        val (A, L, U) = generateForLU(4)
        println("\nLU-разложение ${it+1}\n")

        print("Задача: А ")
        println(A)
        print("Ответ: L ")
        println(L)
        print("Ответ: U ")
        println(U)
    }
}

fun demoCholesky() {
    // исходная матрица
    val A = matrix(
        1,  2,  3,  4,
        2,  5,  7,  3,
        3,  7, 14,  1,
        4,  3,  1, 59
    )
    println(A)
    val L = A.cholesky()
    print("L ")
    println(L)
    print("L^T ")
    println(L.transpose())
}

fun demoLU() {
    // исходная матрица
    val A = matrix(
        4, 8, 12, -4, 2,
        8, 20, 12, -6, 0,
        -8, -8, -39, 18, -21,
        8, 12, 54, 3, -9,
        -4, 0, -63, -9, 19
    )
    println(A)
    val (L, U) = A.decomposeLU()
    print("L ")
    println(L)
    print("U ")
    println(U)
}
