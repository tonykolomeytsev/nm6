class Complex(val re: Float = 0f, val im: Float = 0f) : Number() {

    constructor(re: Number, im: Number) : this(re.toFloat(), im.toFloat())

    override fun toByte() = re.toByte()

    override fun toChar() = re.toChar()

    override fun toDouble() = re.toDouble()

    override fun toFloat() = re

    override fun toInt() = re.toInt()

    override fun toLong() = re.toLong()

    override fun toShort() = re.toShort()

    operator fun plus(another: Number) = if (another is Complex)
        Complex(re + another.re, im + another.im)
    else
        Complex(re + another.toFloat(), im)


    operator fun minus(another: Number) = if (another is Complex)
        Complex(re - another.re, im - another.im)
    else
        Complex(re - another.toFloat(), im)


    operator fun times(another: Number) = if (another is Complex)
        Complex(re*another.re - im*another.im, re*another.im + im*another.re)
    else
        Complex(re * another.toFloat(), im * another.toFloat())

    override fun toString() = "($re + ${im}i)"
}