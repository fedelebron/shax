# shax

This is a simple implementation of the tracing + linearization + transposition approach to automatic differentiation, as explained in the [You Only Linearize Once paper](https://arxiv.org/abs/2204.10923).

This approach uses an EDSL a-la JAX, instead of [a previous approach I tried](https://github.com/fedelebron/hautodiff) which was a compiler.

So far this implements reverse-mode automatic differentiation and a very basic gradient descent, to show how we can go from polymorphic Haskell functions to their gradients and then their minimums. These Haskell functions take types that follow both the `Floating` typeclass, for basic arithmetic operations, and a typeclass mimicing some parts of the Numpy API (dot products, broadcasts, etc). This allows for tracing (a-la JAX's `Tracer`s), but also constant evaluation when given explicit arrays.