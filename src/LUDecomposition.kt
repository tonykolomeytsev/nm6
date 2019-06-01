import kotlin.math.*
import kotlin.random.Random

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

/**
 * Матрица заполненная единицами
 */
fun ones(size: Size) = Array(size.first) { Array(size.second) { 1f }  }

/**
 * Матрица заполненная нулями
 */
fun zeros(size: Size) = Array(size.first) { Array(size.second) { 0f }  }

/**
 * Единичная матрица
 */
fun eye(size: Size) = Array(size.first) { i -> Array(size.second) { j -> if (i == j) 1f else 0f }  }

/**
 * Матрица заполненная числами из диапазона
 */
fun randomRange(size: Size, range: IntRange): Matrix {
    val digits = range.asSequence().toList()
    return matrix(size) { _,_-> digits.random().toFloat() }
}

/**
 * Матрица заполненная рандомными числами от 0 до 1
 */
fun random(size: Size): Matrix {
    val r = Random(System.currentTimeMillis())
    return matrix(size) { _,_-> r.nextFloat() }
}

/**
 * Диагональная матрица
 */
fun diag(vararg numbers: Number, size: Size? = null): Matrix {
    val s = size ?: Size(numbers.size, numbers.size)
    val m = zeros(s)
    for (i in 0 until numbers.size) {
        m[i, i] = numbers[i].toFloat()
    }
    return m
}

fun copy(matrix: Matrix): Matrix {
    val size = matrix.size()
    return Array(size.first) { i -> Array(size.second) { j -> matrix[i, j] }  }
}

/**
 * Размер матрицы
 */
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

    if (A.size == 1) return A.toFloat()

    if (A.size == 2) return A[0, 0]*A[1, 1]-A[0, 1]*A[1, 0]

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

/**
 * Минор
 */
fun Matrix.minor(i_: Int, j_: Int): Float {
    if (!isSquare()) throw ArithmeticException("Non-square matrix")
    val s = Size(size - 1, size - 1)
    val M = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            M[i, j] = this[if (i < i_) i else i + 1, if (j < j_) j else j + 1]
        }
    }
    return det(M)
}

/**
 * Алгебраическое дополнение (кофактор)
 */
fun Matrix.cofactor(i_: Int, j_: Int): Float {
    if (!isSquare()) throw ArithmeticException("Non-square matrix")
    val s = Size(size - 1, size - 1)
    val M = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            M[i, j] = this[if (i < i_) i else i + 1, if (j < j_) j else j + 1]
        }
    }
    return det(M) * (if ((i_+j_) % 2 == 0) 1f else -1f)
}

/**
 * Присоединенная матрица (матрица кофакторов)
 */
fun Matrix.cofactorMatrix(): Matrix = matrix(size()) { i, j -> this.cofactor(i,j) }

/**
 * Обратная матрица
 */
fun Matrix.inversed(): Matrix {
    val d = det(this)
    if (d == 0f) throw ArithmeticException("Det = 0, inverse matrix does not exist")
    return (this.cofactorMatrix() / d)
}

/**
 * Вычисление норм матрицы
 */
fun norm(matrix: Matrix, type: Any? = "inf"): Float {
    when (type) {
        1 -> {
            var max = 0f
            for (j in 0 until matrix[0].size) {
                var sum = 0f
                for (i in 0 until matrix.size)
                    sum += abs(matrix[i ,j])
                if (sum > max) max = sum
            }
            return max
        }
        2 -> {
            TODO("Spectral norma")
        }
        "inf" -> {
            var max = 0f
            for (i in 0 until matrix.size) {
                var sum = 0f
                for (j in 0 until matrix[0].size)
                    sum += abs(matrix[i ,j])
                if (sum > max) max = sum
            }
            return max
        }
    }
    throw ArithmeticException("Undefined norma type")
}

/**
 * Обусловленность матрицы
 */
fun cond(matrix: Matrix): Float = norm(matrix) * norm(matrix.inversed())
fun Matrix.condition() = cond(this)

/**
 * Степень обусловленности матрицы
 */
fun dcond(matrix: Matrix): Float = 1f / cond(matrix)
fun Matrix.conditionDegree() = dcond(this)

fun Matrix.givensRotation(): Matrix {
    val A = copy(this)
    for (i in 0 until (size - 1)) {
        for (j in (i+2) until size)
        {
            val t = 2*A[i, j] / (A[i, i] - A[j, j])
            val phi = 0.5 * atan(t)
            val c = cos(phi)
            val s = sin(phi)

            val bii = c*c* A[i, i] + 2*c*s*A[i, j] + s*s*A[j, j]
            val bij = s*c*(A[j, j] - A[i, i]) + A[i, j] * (c*c - s*s)
            val bjj = s*s*A[i, i] + c*c*A[j, j] - 2*c*s*A[i, j]
            val bji = bij

            A[i, i] = bii
            A[i, j] = bij
            A[j, i] = bji
            A[j, j] = bjj
        }
    }
    return A
}

fun Matrix.filtered(f: Float = 10e-5f) = matrix(size()) { i, j -> if (this[i, j] < f) 0f else this[i, j] }

/**
 * Проверка на нижнетреугольность
 */
fun Matrix.isLowerTriangle(): Boolean {
    val s = size()
    var i = 0
    var j: Int
    while (i < s.first - 1) {
        j = i+1
        while (j < s.second) {
            if (this[i, j] != 0f) return false
            j++
        }
        i++
    }
    return true
}

/**
 * Проверка на верхнетреугольность
 */
fun Matrix.isUpperTriangle(): Boolean {
    val s = size()
    var i = 0
    var j: Int
    while (i < s.first - 1) {
        j = i+1
        while (j < s.second) {
            if (this[j, i] != 0f) return false
            j++
        }
        i++
    }
    return true
}

/**
 * Проверка на диагональность
 */
fun Matrix.isDiagonal(): Boolean {
    val s = size()
    var i = 0
    var j: Int
    while (i < s.first) {
        j = 0
        while (j < s.second) {
            if (this[j, i] != 0f && j != i) return false
            j++
        }
        i++
    }
    return true
}

/**
 * Проверка на симметричность
 */
fun Matrix.isSymmetric(): Boolean {
    val s = size()
    var i = 0
    var j: Int
    while (i < s.first - 1) {
        j = i+1
        while (j < s.second) {
            if (this[i, j] != this[j, i]) return false
            j++
        }
        i++
    }
    return true
}

/**
 * Проверка на ортогональность по необходимому и достаточному условию
 */
fun Matrix.isOrthogonal(): Boolean = matmul(this.transposed(), this).isEye()

/**
 * Проверка на единичность
 */
fun Matrix.isEye(): Boolean {
    var i = 0
    val s= size().min()
    while (i < s) {
        if (this[i, i] != 1f) return false
        i++
    }
    return isDiagonal()
}

/**
 * Сравнение двух матриц
 */
fun Matrix.equalsMatrix(another: Matrix): Boolean {
    val s = size()
    var i = 0
    var j: Int
    while (i < s.first) {
        j = 0
        while (j < s.second) {
            if (this[i, j] != another[i, j]) return false
            j++
        }
        i++
    }
    return true
}

/**
 * Приведение к скаляру матрицы размера 1х1
 */
fun Matrix.toFloat(): Float {
    if (size() != Size(1,1)) throw ArithmeticException("Matrix is not a scalar (size != 1x1)")
    return this[0, 0]
}

/**
 * Проверка на квадратность
 */
fun Matrix.isSquare() = (size == this[0].size)

/**
 * Минимальный размер матрицы
 */
fun Size.min() = (if (first > second) second else first)

/**
 * Максимальный размер матрицы
 */
fun Size.max() = (if (first > second) first else second)

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

/**
 * Поэлементное умножение (для матричного умножения используй matmul(A,B)
 */
operator fun Matrix.times(another: Matrix): Matrix {
    if (this.size() != another.size()) throw ArithmeticException("Matrix sizes does not equal")
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] * another[i, j]
        }
    }
    return C
}

/**
 * Поэлементное деление
 */
operator fun Matrix.div(another: Matrix): Matrix {
    if (this.size() != another.size()) throw ArithmeticException("Matrix sizes does not equal")
    val s = size()
    val C = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            C[i, j] = this[i, j] / another[i, j]
        }
    }
    return C
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
        1, 2, 3, 4,
        0, 5, 6, 7,
        0, 0, 9, 10,
        11, 12, 13, 14
    )
    val B = matrix(
        1, 10, 100, 1001
    )
    val C = diag(1, 5, 9)

    val D = matrix(
        0.8, -0.6,
        -0.6, -0.8
    )
    println(A.givensRotation().filtered())
}
