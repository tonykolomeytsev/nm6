import kotlin.math.*
import kotlin.random.Random

/**
 * Матрица это массив из массивов вещественных чисел
 */
typealias Matrix = Array<Array<Float>>
typealias Vector = Array<Float>
typealias Size = Pair<Int, Int>

/**
 * Конструктор для создания матрицы произвольной формы
 */
fun matrix(vararg e: Number, size: Size? = null): Matrix {
    val s = size ?: { i: Int -> Size(i, i) }(sqrt(e.size.toFloat()).toInt())
    var i = 0
    return Array(s.first) { Array(s.second) {e[i++].toFloat()} }
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
fun random(size: Size, seed: Long? = null): Matrix {
    val r = Random(seed ?: System.currentTimeMillis())
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

fun tril(size: Size) = matrix(size) { i, j -> if (i >= j) 1f else 0f }

fun triu(size: Size) = matrix(size) { i, j -> if (i <= j) 1f else 0f }

/**
 * Копия матрицы
 */
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

fun Matrix.column(j: Int) = Vector(size) { i -> this[i, j] }

fun Matrix.row(i: Int): Vector = this[i]

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
 * Минор (определитель матрицы-минора)
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
 * Матрица-минор без вычисления определителя
 */
fun Matrix.minorMatrix(i_: Int, j_: Int): Matrix {
    if (!isSquare()) throw ArithmeticException("Non-square matrix")
    val s = Size(size - 1, size - 1)
    val M = zeros(s)
    for (i in 0 until s.first) {
        for (j in 0 until s.second) {
            M[i, j] = this[if (i < i_) i else i + 1, if (j < j_) j else j + 1]
        }
    }
    return M
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



fun Matrix.floor() = Array(size) { i -> Array(this[0].size) { j -> floor(this[i, j]) } }

fun Matrix.ceil() = Array(size) { i -> Array(this[0].size) { j -> ceil(this[i, j]) } }

fun Matrix.round() = Array(size) { i -> Array(this[0].size) { j -> round(this[i, j]) } }

fun Matrix.sqrt() = Array(size) { i -> Array(this[0].size) { j -> sqrt(this[i, j]) } }

fun Matrix.sin() = Array(size) { i -> Array(this[0].size) { j -> sin(this[i, j]) } }

fun Matrix.cos() = Array(size) { i -> Array(this[0].size) { j -> cos(this[i, j]) } }

fun Matrix.pow(power: Float) = Array(size) { i -> Array(this[0].size) { j -> Math.pow(this[i, j].toDouble(), power.toDouble()).toFloat() } }

fun Matrix.matMap(function: (Float) -> Float) = Array(size) { i -> Array(this[0].size) { j -> function(this[i, j]) } }

fun Matrix.matMap(function: (Int, Int, Float) -> Float) = Array(size) { i -> Array(this[0].size) { j -> function(i, j, this[i, j]) } }

fun Matrix.matForEach(function: (Int, Int, Float) -> Unit) {
    val s = size()
    for (i in 0 until s.first)
        for (j in 0 until s.second)
            function(i, j, this[i, j])
}

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
    if (size() != Size(1,1)) throw ArithmeticException("Matrix is not re scalar (size != 1x1)")
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

operator fun Number.times(another: Vector): Vector = Vector(another.size) { i -> another[i] * this.toFloat() }

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

operator fun Vector.div(another: Vector): Vector = this.mapIndexed { i, e -> e / another[i] }.toTypedArray()

operator fun Vector.div(another: Number): Vector = this.map { e -> e / another.toFloat() }.toTypedArray()

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

operator fun Vector.plus(another: Vector): Vector = this.mapIndexed { i, e -> e + another[i] }.toTypedArray()

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

operator fun Vector.minus(another: Vector): Vector = this.mapIndexed { i, e -> e - another[i] }.toTypedArray()

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


fun Vector.norm(type: Any? = 2): Float {
    when (type) {
        1 -> return this.reduce { acc, fl -> acc + abs(fl) }
        2 -> return sqrt(this.reduce { acc, fl -> acc + fl*fl })
        "inf" -> return this.maxBy { abs(it) }!!
    }
    throw ArithmeticException("Undefined norma type")
}

/**
 * QR-разложение, возвращает Q, R
 */
fun Matrix.decomposeQR(): Pair<Matrix, Matrix> {
    val compute_minor = { mat: Matrix, d: Int ->
        val A = zeros(mat.size())
        for (i in 0 until d)
            A[i, i] = 1f
        for (i in d until mat.size)
            for (j in d until mat[0].size)
                A[i, j] = mat[i, j]
        A
    }
    val sign = { a: Float -> if (a > 0) 1f else -1f }

    val n = size().first
    val m = size().second

    val qv = Array<Matrix?>(m) { null }
    var z = copy(this)

    var k = 0
    while (k < n && k < m - 1) {
        val a = z.column(k)
        val e = Vector(n) { i -> if (i == k) 1f else 0f }
        val u = a + (sign(a[0]) * a.norm() * e)
        val v = u / u[0]
        val vTv = v.reduce { acc, fl -> acc + fl*fl }

        // compute householder factor Q = I - 2*v*v^T
        qv[k] = eye(Size(n, n)) - 2*matrix(Size(n, n)) { i, j -> v[i] * v[j] } / vTv

        z = matmul(qv[k]!!, z)
        k++
    }

    var Q = qv[0]!!

    var i = 1
    while (i < n && i < m - 1) {
        Q = matmul(qv[i]!!, Q)
        i++
    }

    val R = matmul(Q, this)
    return Pair(Q.transposed(), R)
}

/**
 * Возвращает U, S, V (транспонируешь сам)
 */
fun Matrix.decomposeSVD(): Triple<Matrix, Matrix, Matrix> {
    val s = size()
    val A = this
    val H = zeros(s)// Householder matrix
    val U = zeros(Size(s.first, s.first))
    val S = zeros(s)
    val V = zeros(Size(s.second, s.second))
    return Triple(U, S, V)
}


