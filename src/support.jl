typed_decompose(x::T) where {T} = (T, Base.decompose(x)...)
  
function typed_recompose(::Type{T}, significand::I, exponent::I, sign::I) where {T, I<:Integer}
    ldexp(copysign(convert(T, significand), sign), exponent)
end


"""
    classify_float(x)

x             --> 0
iszero(x)     --> 1
isinf(x)      --> 2
isnan(x)      --> 3

The class identifier c can be readily computed in a branchfree
manner out of the following pieces of information that
can be accessed with masking and bit manipulation [1]:
- the biased exponent bits are all zeros (x is zero or subnormal),
- the biased exponent bits are all ones (x is infinity or NaN),
- the IEEE754 significand bits are all zeros (x is zero or infinity)
"""
@inline function classify_float(x)
   iszero(x) + isinf(x) << 1 + 3 * isnan(x)
end

const IntXp = NamedTuple{(:class, :significand, :exponent, :sign), Tuple{Int,Int,Int,Int}}

"""
    intxp(x)

similar to frexp(x) with integer results

sign is +1 or -1

significand is an integer, normalized such that 2^(precision - 1) <= significand <= 2^(precision) - 1
- whether x is normal or subnormal or special

exponent makes copysign(ldexp(T(significand), exponent), sign) == x::T

intxp(1.0f0) == (class = 0, significand = 8388608, exponent = -23, sign = 1)


"""
@inline intxp(x) = IntXp( (classify_float(x), Base.decompose(x)...) )
    

#=
   ref: https://graphics.stanford.edu/~seander/bithacks.html#IntegerMinOrMax

Compute the minimum (min) or maximum (max) of two integers without branching
int x;  // we want to find the minimum of x and y
int y;   
int r;  // the result goes here 

To find the minimum, use:
r = y ^ ((x ^ y) & -(x < y)); // min(x, y)

On some rare machines where branching is very expensive and no condition move instructions exist,
the above expression might be faster than the obvious approach, r = (x < y) ? x : y, even though
it involves two more instructions. (Typically, the obvious approach is best, though.) 
It works because if x < y, then -(x < y) will be all ones, so r = y ^ (x ^ y) & ~0 = y ^ x ^ y = x.
Otherwise, if x >= y, then -(x < y) will be all zeros, so r = y ^ ((x ^ y) & 0) = y. 
On some machines, evaluating (x < y) as 0 or 1 requires a branch instruction, so there may be no advantage.

To find the maximum, use:
r = x ^ ((x ^ y) & -(x < y)); // max(x, y)

Quick and dirty versions:
If you know that INT_MIN <= x - y <= INT_MAX, then you can use the following, which are faster
because (x - y) only needs to be evaluated once.

r = y + ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))); // min(x, y)
r = x - ((x - y) & ((x - y) >> (sizeof(int) * CHAR_BIT - 1))); // max(x, y)

Note that the 1989 ANSI C specification doesn't specify the result of signed right-shift,
so these aren't portable. If exceptions are thrown on overflows, then the values of x and y
should be unsigned or cast to unsigned for the subtractions to avoid unnecessarily throwing 
an exception, however the right-shift needs a signed operand to produce all one bits when negative,
so cast to signed there.

On March 7, 2003, Angus Duggan pointed out the right-shift portability issue. 
On May 3, 2005, Randal E. Bryant alerted me to the need for the precondition, INT_MIN <= x - y <= INT_MAX,
and suggested the non-quick and dirty version as a fix. 
Both of these issues concern only the quick and dirty version. 

Nigel Horspoon observed on July 6, 2005 that gcc produced the same code on a Pentium as the obvious solution
because of how it evaluates (x < y). 
On July 9, 2008 Vincent Lefèvre pointed out the potential for overflow exceptions with subtractions in
r = y + ((x - y) & -(x < y)), which was the previous version. 
Timothy B. Terriberry suggested using xor rather than add and subract to avoid casting and the risk of overflows on June 2, 2009.

=#

@inline min_max_internal(x::I, y::I) where {I<:Integer} = ((x ⊻ y) & -(x < y))

@inline min_branchless(x::I, y::I) where {I<:Integer} = y ⊻ min_max_internal(x, y)
@inlnie max_branchless(x::I, y::I) where {I<:Integer} = x ⊻ min_max_internal(x, y)

@inline function minmax_branchless(x::I, y::I) where {I<:Integer}
    mm = min_max_internal(x, y)
    y ⊻ mm, x ⊻ mm
end

@inline function maxmin_branchless(x::I, y::I) where {I<:Integer}
    mm = min_max_internal(x, y)
    x ⊻ mm, y ⊻ mm
end


