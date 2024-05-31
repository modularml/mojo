# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Defines Quaternion, a structure often used to represent rotations."""

from math import sqrt, sin, cos


# ===----------------------------------------------------------------------===#
# Quaternion
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct Quaternion[T: DType = DType.float64]:
    """A Quaternion allocated on the stack with very efficient vectorized
    operations.

    Parameters:
        T: The type of the elements in the Quaternion, must be a
            floating point type.
    """

    alias _vec_type = SIMD[T, 4]
    alias _scalar_type = Scalar[T]
    var vec: Self._vec_type
    """The underlying SIMD vector."""

    fn __init__(
        inout self,
        w: Self._scalar_type = 1,
        i: Self._scalar_type = 0,
        j: Self._scalar_type = 0,
        k: Self._scalar_type = 0,
    ):
        """Construct a Quaternion from a real and an imaginary vector part.

        Args:
            w: Real part.
            i: Imaginary i, equivalent to vector part x.
            j: Imaginary j, equivalent to vector part y.
            k: Imaginary k, equivalent to vector part z.
        """
        constrained[
            T.is_floating_point(),
            msg="Quaternions can only be expressed with floating point types",
        ]()
        self.vec = Self._vec_type(w, i, j, k)

    fn __init__(inout self, vec: Self._vec_type):
        """Construct a Quaternion from a real and an imaginary vector part.

        Args:
            vec: A SIMD vector representing the Quaternion.
        """
        constrained[
            T.is_floating_point(),
            msg="Quaternions can only be expressed with floating point types",
        ]()
        self.vec = vec

    fn __init__(
        inout self,
        *,
        x: Self._scalar_type,
        y: Self._scalar_type,
        z: Self._scalar_type,
        theta: Self._scalar_type,
        is_normalized: Bool = False,
    ):
        """Construct a Quaternion from a rotation axis vector and an angle.

        Args:
            x: Vector part x.
            y: Vector part y.
            z: Vector part z.
            theta: Rotation angle.
            is_normalized: Whether the input vector is normalized.
        """
        constrained[
            T.is_floating_point(),
            msg="Quaternions can only be expressed with floating point types",
        ]()
        var vec = Self._vec_type(0, x, y, z)
        if not is_normalized:
            vec = vec / sqrt((vec**2).reduce_add())

        var sin_a = sin(theta * 0.5)
        var cos_a = cos(theta * 0.5)
        self.vec = vec * sin_a
        self.vec[0] = cos_a
        self.normalize()

    fn __abs__(self) -> Self._scalar_type:
        """Get the magnitude of the Quaternion.

        Returns:
            The magnitude.
        """
        return sqrt((self.vec**2).reduce_add())

    fn normalize(inout self):
        """Normalize the Quaternion."""
        self.vec /= self.__abs__()

    fn __getattr__[name: StringLiteral](self) -> Self._scalar_type:
        """Get the attribute.

        Parameters:
            name: The name of the attribute: {"w", "i", "j", "k"}.

        Returns:
            The attribute value.
        """

        @parameter
        if name == "w":
            return self.vec[0]
        elif name == "i":
            return self.vec[1]
        elif name == "j":
            return self.vec[2]
        elif name == "k":
            return self.vec[3]
        else:
            constrained[False, msg="that attribute isn't defined"]()
            return 0

    fn conjugate(self) -> Self:
        """Return the conjugate of the Quaternion.

        Returns:
            The conjugate.
        """
        return Self(self.vec * Self._vec_type(1, -1, -1, -1))

    fn __invert__(self) -> Self:
        """Return the conjugate of the Quaternion.

        Returns:
            The conjugate.
        """
        return self.conjugate()

    fn dot(self, other: Self) -> Self._scalar_type:
        """Calculate the dot product of self with other.

        Args:
            other: The other Quaternion.

        Returns:
            The result.
        """
        return (self.vec * other.vec).reduce_add()

    fn __mul__(self, other: Self) -> Self:
        """Calculate the Hamilton product of self with other.

        Returns:
            The result.
        """
        var w = self.dot(~other)
        var i = self.dot(Self._vec_type(other.i, other.w, other.k, -other.j))
        var j = self.dot(Self._vec_type(other.j, -other.k, other.w, other.i))
        var k = self.dot(Self._vec_type(other.k, other.j, -other.i, other.w))
        return Self(w, i, j, k)

    fn __imul__(inout self, other: Self):
        """Calculate the Hamilton product of self with other inplace.

        Args:
            other: The other Quaternion.
        """
        self = self * other

    fn __truediv__(self, other: Self) -> Self:
        """Calculate the division of self with other.

        Args:
            other: The other Quaternion.

        Returns:
            The result.
        """
        return self * ~other

    fn __itruediv__(inout self, other: Self):
        """Calculate the division of self with other inplace.

        Args:
            other: The other Quaternion.
        """
        self = self / other

    fn sqrt(self) -> Self:
        """Calculate the square root of the Quaternion.

        Returns:
            The result.
        """

        var vec_magn = (
            (self.vec * Self._vec_type(0, 1, 1, 1)) ** 2
        ).reduce_add()
        var vec = self.vec * (sqrt((self.__abs__() - self.w) / 2) / vec_magn)
        return Self(sqrt((self.__abs__() + self.w) / 2), vec[1], vec[2], vec[3])

    # fn to_matrix(self) -> Matrix[T, 3, 3]:
    #     """Calculate the 3x3 rotation Matrix from the Quaternion.

    #     Returns:
    #         The resulting 3x3 Matrix.
    #     """
    #     var vec = (~self).vec
    #     var wxyz_x = vec * vec[1]
    #     var wxyz_y = vec * vec[2]
    #     var wxyz_z = vec * vec[3]
    #     var wx = wxyz_x[0]
    #     var xx = wxyz_x[1]
    #     var yx = wxyz_x[2]
    #     var zx = wxyz_x[3]

    #     var wy = wxyz_y[0]
    #     var xy = wxyz_y[1]
    #     var yy = wxyz_y[2]
    #     var zy = wxyz_y[3]

    #     var wz = wxyz_z[0]
    #     var xz = wxyz_z[1]
    #     var yz = wxyz_z[2]
    #     var zz = wxyz_z[3]

    #     var mat = Matrix[T, 3, 3](
    #         1 - 2 * (yy + zz),
    #         2 * (xy + wz),
    #         2 * (zx - wy),
    #         2 * (xy - wz),
    #         1 - 2 * (xx + zz),
    #         2 * (zy + wz),
    #         2 * (zx + wy),
    #         2 * (zy - wz),
    #         1 - 2 * (xx + yy),
    #     )
    #     return mat
