import kotlin.math.sqrt

/**
 * Матрица это массив из массивов вещественных чисел
 */
typealias Matrix = Array<Array<Float>>
typealias Size = Pair<Int, Int>
/**
 * Конструктор для упрощенного создания квадратной матрицы
 */
fun matrix(vararg e: Number): Matrix {
    val size = sqrt(e.size.toFloat()).toInt()
    var i = 0
    return Array(size) { Array(size) {e[i++].toFloat()} }
}

/**
 * Конструктор для создания матрицы произвольной формы
 */
fun matrix(size: Size, vararg e: Number): Matrix {
    var i = 0
    return Array(size.first) { Array(size.second) {e[i++].toFloat()} }
}

/**
 * Конструктор для создания матрицы, заполненной указанными значениями зависящими от i, j
 */
fun matrix(size: Size, lambda: (Int, Int) -> Float = { _,_-> 0f })
        = Array(size.first) { i -> Array(size.second) { j -> lambda(i, j) }  }

/**
 * Конструктор для создания матрицы, заполненной указанными константами
 */
fun matrix(size: Size, lambda: () -> Float = { 0f })
        = Array(size.first) { Array(size.second) { lambda() }  }

fun ones(size: Size) = Array(size.first) { Array(size.second) { 1f }  }

fun zeros(size: Size) = Array(size.first) { Array(size.second) { 0f }  }

fun Matrix.size() = Size(size, this[0].size)

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

// арифметика

operator fun Number.plus(another: Matrix): Matrix {
    val size = another.size()
    val C = zeros(size)
    for (i in 0 until size.first) {
        for (j in 0 until size.second) {
            C[i, j] = another[i, j] + this.toFloat()
        }
    }
    return C
}

operator fun Number.minus(another: Matrix): Matrix {
    val size = another.size()
    val C = zeros(size)
    for (i in 0 until size.first) {
        for (j in 0 until size.second) {
            C[i, j] = another[i, j] - this.toFloat()
        }
    }
    return C
}

operator fun Number.times(another: Matrix): Matrix {
    val size = another.size()
    val C = zeros(size)
    for (i in 0 until size.first) {
        for (j in 0 until size.second) {
            C[i, j] = another[i, j] * this.toFloat()
        }
    }
    return C
}

operator fun Number.div(another: Matrix): Matrix {
    val size = another.size()
    val C = zeros(size)
    for (i in 0 until size.first) {
        for (j in 0 until size.second) {
            C[i, j] = another[i, j] / this.toFloat()
        }
    }
    return C
}

operator fun Number.rem(another: Matrix): Matrix {
    val size = another.size()
    val C = zeros(size)
    for (i in 0 until size.first) {
        for (j in 0 until size.second) {
            C[i, j] = another[i, j] % this.toFloat()
        }
    }
    return C
}

operator fun Matrix.times(another: Number): Matrix {
    val C = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            C[i, j] = this[i, j] * another.toFloat()
        }
    }
    return C
}

operator fun Matrix.timesAssign(another: Number) {
    for (i in 0 until size) {
        for (j in 0 until size) {
            this[i, j] *= another.toFloat()
        }
    }
}

operator fun Matrix.div(another: Number): Matrix {
    val C = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            C[i, j] = this[i, j] / another.toFloat()
        }
    }
    return C
}

operator fun Matrix.divAssign(another: Number) {
    for (i in 0 until size) {
        for (j in 0 until size) {
            this[i, j] /= another.toFloat()
        }
    }
}

operator fun Matrix.rem(another: Number): Matrix {
    val C = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            C[i, j] = this[i, j] % another.toFloat()
        }
    }
    return C
}

operator fun Matrix.remAssign(another: Number) {
    for (i in 0 until size) {
        for (j in 0 until size) {
            this[i, j] %= another.toFloat()
        }
    }
}

operator fun Matrix.plus(another: Number): Matrix {
    val C = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            C[i, j] = this[i, j] + another.toFloat()
        }
    }
    return C
}

operator fun Matrix.plus(another: Matrix): Matrix {
    val C = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            C[i, j] = this[i, j] + another[i, j]
        }
    }
    return C
}

operator fun Matrix.minus(another: Number): Matrix {
    val C = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            C[i, j] = this[i, j] - another.toFloat()
        }
    }
    return C
}

operator fun Matrix.minus(another: Matrix): Matrix {
    val C = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            C[i, j] = this[i, j] - another[i, j]
        }
    }
    return C
}

operator fun Matrix.plusAssign(another: Number) {
    for (i in 0 until size) {
        for (j in 0 until size) {
            this[i, j] += another.toFloat()
        }
    }
}

operator fun Matrix.plusAssign(another: Matrix) {
    for (i in 0 until size) {
        for (j in 0 until size) {
            this[i, j] += another[i, j]
        }
    }
}

operator fun Matrix.minusAssign(another: Number) {
    for (i in 0 until size) {
        for (j in 0 until size) {
            this[i, j] -= another.toFloat()
        }
    }
}

operator fun Matrix.minusAssign(another: Matrix) {
    for (i in 0 until size) {
        for (j in 0 until size) {
            this[i, j] -= another[i, j]
        }
    }
}

/**
 * Транспонирование матриц
 */
fun Matrix.transposed(): Matrix {
    val m = Array(size) { Array(size) { 0f } }
    for (i in 0 until size) {
        for (j in 0 until size) {
            m[i, j] = this[j, i]
        }
    }
    return m
}

fun Matrix.transpose() {
    for (i in 0 until size) {
        for (j in i until size) {
            val temp = this[i, j]
            this[i, j] = this[j, i]
            this[j, i] = temp
        }
    }
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

fun print(matrix: Matrix) = print(matrix.asString())

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
fun Matrix.decomposeCholesky(): Matrix {
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
    val A = matmul(L, L.transposed())
    return Pair(A.decomposeCholesky(), A)
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
    val A = matmul(L, U)
    val (testL, testU) = A.decomposeLU()
    return Triple(A, testL, testU)
}


fun main() {
    val A = matrix(
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    )
    println(A)
    A.transpose()
    println(A)
    println(A.transposed())
}
