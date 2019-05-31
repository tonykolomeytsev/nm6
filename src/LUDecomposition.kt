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
    if (A.size().second != B.size().first) throw ArithmeticException("Matrix sizes does not equal")
    val newSize = Size(A.size().first, B.size().second)
    val C = Array(newSize.first) { Array(newSize.second) { 0f } }

    for (i in 0 until newSize.first) {
        for (j in 0 until newSize.second) {
            for (k in 0 until A.size().second) {
                C[i, j] += A[i, k] * B[k, j]
            }
        }
    }

    return C
}

/**
 * Определитель
 */
fun det(A: Matrix): Float {
    if (!A.isSquare()) throw ArithmeticException("Bad matrix size")
    var sumPositive = 0f
    var sumNegative = 0f

    for (i in 0 until A.size) {
        var localMultP = 1f
        var localMultN = 1f
        for (ps in 0 until A.size) {
            val iCoord = (i + ps) % A.size
            localMultP *= A[iCoord, ps]
            localMultN *= A[iCoord, A.size - ps - 1]
        }
        sumPositive += localMultP
        sumNegative += localMultN
    }

    return sumPositive - sumNegative
}

fun Matrix.determinant() = det(this)

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

// еще арифметика

operator fun Matrix.times(another: Number): Matrix {
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] * another.toFloat()
        }
    }
    return C
}

operator fun Matrix.timesAssign(another: Number) {
    val s = size()
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            this[i, j] *= another.toFloat()
        }
    }
}

operator fun Matrix.div(another: Number): Matrix {
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] / another.toFloat()
        }
    }
    return C
}

operator fun Matrix.divAssign(another: Number) {
    val s = size()
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            this[i, j] /= another.toFloat()
        }
    }
}

operator fun Matrix.rem(another: Number): Matrix {
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] % another.toFloat()
        }
    }
    return C
}

operator fun Matrix.remAssign(another: Number) {
    val s = size()
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            this[i, j] %= another.toFloat()
        }
    }
}

operator fun Matrix.plus(another: Number): Matrix {
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] + another.toFloat()
        }
    }
    return C
}

operator fun Matrix.plus(another: Matrix): Matrix {
    if (this.size() != another.size()) throw ArithmeticException("Matrix sizes does not equal")
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] + another[i, j]
        }
    }
    return C
}

operator fun Matrix.minus(another: Number): Matrix {
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] - another.toFloat()
        }
    }
    return C
}

operator fun Matrix.minus(another: Matrix): Matrix {
    if (this.size() != another.size()) throw ArithmeticException("Matrix sizes does not equal")
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] - another[i, j]
        }
    }
    return C
}

operator fun Matrix.plusAssign(another: Number) {
    val s = size()
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            this[i, j] += another.toFloat()
        }
    }
}

operator fun Matrix.plusAssign(another: Matrix) {
    if (this.size() != another.size()) throw ArithmeticException("Matrix sizes does not equal")
    val s = size()
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            this[i, j] += another[i, j]
        }
    }
}

operator fun Matrix.minusAssign(another: Number) {
    val s = size()
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            this[i, j] -= another.toFloat()
        }
    }
}

operator fun Matrix.minusAssign(another: Matrix) {
    if (this.size() != another.size()) throw ArithmeticException("Matrix sizes does not equal")
    val s = size()
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            this[i, j] -= another[i, j]
        }
    }
}

/**
 * Транспонирование матриц
 */
fun Matrix.transposed(): Matrix {
    val newSize = Size(size().second, size().first)
    val m = zeros(newSize)
    for (i in 0 until newSize.first) {
        for (j in 0 until newSize.second) {
            m[i, j] = this[j, i]
        }
    }
    return m
}

fun Matrix.transpose() {
    if (size().first != size().second) throw ArithmeticException("In-place transpose only for square matrix")
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
    val z = size()
    var s = "Matrix (${z.first}x${z.second}):\n"
    for (i in 0 until z.first) {
        for (j in 0 until z.second) {
            val x = String.format("%1$10s", this[i, j])
            s += "$x "
        }
        s+= "\n"
    }
    return s
}

fun println(matrix: Matrix) = println(matrix.asString())

fun print(matrix: Matrix) = print(matrix.asString())

fun Matrix.toFloat(): Float {
    if (size() != Size(1,1)) throw ArithmeticException("Matrix is not a scalar (size != 1x1)")
    return this[0, 0]
}

fun Matrix.isSquare() = (size == this[0].size)

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



fun main() {
    val A = matrix(
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    )
    println(A )
    println(det(A))
}
