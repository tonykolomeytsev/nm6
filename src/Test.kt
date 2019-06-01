

fun main() {
    val A = matrix(
        12,  -51,   4,
         6,  167, -68,
        -4,   24, -41,
        -1,    1,   0,
         2,    0,   3,
        size = Size(5, 3)
    )
    print("A ")
    println(A)

    val (Q, R) = A.decomposeQR()
    print("Q ")
    println(Q)
    print("R ")
    println(R)

    print("A == Q*R ? ")
    println(matmul(Q, R).filtered())

}
