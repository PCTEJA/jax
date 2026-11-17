# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Precision tests for GELU against reference implementations."""

from functools import partial

from absl.testing import parameterized
import jax
from jax._src import config
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.tests.numerics import numerics_test_util as util
import mpmath
import numpy as np
import scipy.special

config.parse_flags_with_absl()

bf16, f16, f32, f64 = jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64
DTYPE_PARAMS = [(f"_{d.__name__}", d) for d in [bf16, f16, f32, f64]]

_SQRT_2_OVER_PI = np.sqrt(2 / np.pi)
_CUBIC_COEFF = 0.044715

# Inputs from jax-ml/jax#42641 (float32 gelu(-6) and the bfloat16 values at
# -4, -3, -2) plus the surrounding negative tail, where the approximate
# formulation's 1 + tanh(z) term loses precision, and the positive side.
_INTERESTING_POINTS = [
    -10.0, -8.0, -6.0, -4.87, -4.0, -3.0, -2.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0,
]


def _gelu_exact_ref(x):
  return 0.5 * x * scipy.special.erfc(-x / np.sqrt(2.0))


def _gelu_exact_mpmath(x):
  # Guarded like _mpmath_erfc in erf_test.py: mpmath.erfc overflows for
  # large positive arguments, where the value is 0 to working precision.
  arg = -x / mpmath.sqrt(2)
  erfc = mpmath.mpf(0) if arg > 100 else (
      mpmath.mpf(2) if arg < -100 else mpmath.erfc(arg))
  return x * erfc / 2


def _gelu_approx_ref(x):
  # scipy.special.expit evaluates the same factor as 0.5 * (1 + tanh(z))
  # without the cancellation that the float64 evaluation of 1 + tanh(z)
  # would itself incur for negative z.
  z = _SQRT_2_OVER_PI * (x + _CUBIC_COEFF * x**3)
  return x * scipy.special.expit(2 * z)


def _gelu_approx_mpmath(x):
  # The logistic form is the same factor as 0.5 * (1 + tanh(z)) evaluated
  # without cancellation: at 100-bit precision 1 + tanh(z) is exactly zero
  # for z < -35 even though the true factor is about 1e-32.
  z = mpmath.sqrt(2 / mpmath.pi) * (x + mpmath.mpf("0.044715") * x**3)
  return x / (1 + mpmath.exp(-2 * z))


@jax.jit
def gelu_exact_grad(x):
  return jax.vmap(jax.grad(partial(jax.nn.gelu, approximate=False)))(x)


@jax.jit
def gelu_approx_grad(x):
  return jax.vmap(jax.grad(partial(jax.nn.gelu, approximate=True)))(x)


def _gelu_exact_grad_ref(x):
  return (scipy.special.ndtr(x)
          + x * np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi))


def _gelu_exact_grad_mpmath(x):
  if x < -100:
    return mpmath.mpf(0)
  if x > 100:
    return mpmath.mpf(1)
  return (mpmath.ncdf(x)
          + x * mpmath.exp(-x**2 / 2) / mpmath.sqrt(2 * mpmath.pi))


def _gelu_approx_grad_ref(x):
  z = _SQRT_2_OVER_PI * (x + _CUBIC_COEFF * x**3)
  cdf = scipy.special.expit(2 * z)
  return cdf + x * 2 * cdf * (1 - cdf) * _SQRT_2_OVER_PI * (
      1 + 3 * _CUBIC_COEFF * x**2)


def _gelu_approx_grad_mpmath(x):
  a = mpmath.sqrt(2 / mpmath.pi)
  c = mpmath.mpf("0.044715")
  z = a * (x + c * x**3)
  cdf = 1 / (1 + mpmath.exp(-2 * z))
  return cdf + x / mpmath.cosh(z) ** 2 / 2 * a * (1 + 3 * c * x**2)


@jtu.thread_unsafe_test_class()
class GeluExactTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gelu_exact_accuracy(self, dtype):
    # The float32 spike is at x ~ -13, where erfc underflows to zero and the
    # computed value is -0.0 while the reference is a few times 1e-38, which
    # is millions of subnormal ULPs away.
    bounds = [
        ("cpu", {bf16: 148.0, f16: 15.5, f32: 11427107.5, f64: 1085.5}),
    ]
    util.check_unary_precision(
        self,
        partial(jax.nn.gelu, approximate=False),
        _gelu_exact_ref,
        _gelu_exact_mpmath,
        dtype,
        bounds=bounds,
        interesting_points=_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class GeluApproximateTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gelu_approximate_accuracy(self, dtype):
    # The bounds record the current implementation, whose 0.5 * (1 + tanh(z))
    # factor cancels in the negative tail: for x <= -2 the computed value is
    # exactly zero for a large fraction of inputs even though the reference
    # value is nonzero, which produces the large ULP errors below.
    bounds = [
        ("cpu", {bf16: 256.0, f16: 2037.0, f32: 16777216.5, f64: 4.6e17}),
    ]
    util.check_unary_precision(
        self,
        partial(jax.nn.gelu, approximate=True),
        _gelu_approx_ref,
        _gelu_approx_mpmath,
        dtype,
        bounds=bounds,
        interesting_points=_INTERESTING_POINTS,
    )


@jtu.thread_unsafe_test_class()
class GeluExactGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gelu_exact_grad_accuracy(self, dtype):
    # Same float32 underflow region as in GeluExactTest: at x ~ -13 the
    # computed gradient is 0.0 while the reference is a few times 1e-38.
    bounds = [
        ("cpu", {bf16: 309.5, f16: 1170.0, f32: 12256531.5, f64: 1062.5}),
    ]
    # The gradient at +-inf is NaN, so only finite inputs are checked.
    fmax = float(jnp.finfo(dtype).max)
    util.check_unary_precision(
        self,
        gelu_exact_grad,
        _gelu_exact_grad_ref,
        _gelu_exact_grad_mpmath,
        dtype,
        bounds=bounds,
        interesting_points=_INTERESTING_POINTS,
        domain=lambda dt: util.Domain.interval(dt, -fmax, fmax),
    )


@jtu.thread_unsafe_test_class()
class GeluApproximateGradTest(jtu.JaxTestCase):

  @parameterized.named_parameters(*DTYPE_PARAMS)
  def test_gelu_approximate_grad_accuracy(self, dtype):
    bounds = [
        ("cpu", {bf16: 254.5, f16: 2042.0, f32: 16777216.5, f64: 4.6e17}),
    ]
    # Beyond this limit 0.044715 * x**3 overflows the dtype, z is infinite,
    # and the gradient is NaN (in float16 already at x = 1e4) even though the
    # true gradient is 1. That region is excluded here and reported in
    # jax-ml/jax#42641.
    limit = np.cbrt(float(jnp.finfo(dtype).max)) / np.cbrt(_CUBIC_COEFF)
    util.check_unary_precision(
        self,
        gelu_approx_grad,
        _gelu_approx_grad_ref,
        _gelu_approx_grad_mpmath,
        dtype,
        bounds=bounds,
        interesting_points=_INTERESTING_POINTS,
        domain=lambda dt: util.Domain.interval(dt, -limit, limit),
    )


util.register_benchmark(partial(jax.nn.gelu, approximate=False),
                        name="gelu_exact")
util.register_benchmark(partial(jax.nn.gelu, approximate=True),
                        name="gelu_approximate")
util.register_benchmark(gelu_exact_grad)
util.register_benchmark(gelu_approx_grad)


if __name__ == "__main__":
  util.main()
