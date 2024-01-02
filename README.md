# shax

This is a simple implementation of the tracing + linearization + transposition approach to automatic differentiation, as explained in the [You Only Linearize Once paper](https://arxiv.org/abs/2204.10923).

This approach uses an EDSL a-la JAX, instead of [a previous approach I tried](https://github.com/fedelebron/hautodiff) which was a compiler.

So far this implements reverse-mode automatic differentiation and a very basic gradient descent, to show how we can go from polymorphic Haskell functions to their gradients and then their minimums. These Haskell functions take types that follow both the `Floating` typeclass, for basic arithmetic operations, and a typeclass mimicing some parts of the Numpy API (dot products, broadcasts, etc). This allows for tracing (a-la JAX's `Tracer`s), but also constant evaluation when given explicit arrays.

For example, when given the following function:

```haskell
f :: forall a. Floating a => a -> a -> a
f x y = let z = x + y
        in sin z * cos z
```

Running gradient descent on this does:

```
$ stack build && clear && stack exec shax-exe
def f(arg0 :: f32[], 
      arg1 :: f32[]) =
  let x0 :: f32[] = arg0
      x1 :: f32[] = arg1
      x2 :: f32[] = add x0 x1
      x3 :: f32[] = sin x2
      x4 :: f32[] = cos x2
      x5 :: f32[] = mul x3 x4
  in [x5]
Iteration 0: -0.23971277
Iteration 1: -0.31265023
Iteration 2: -0.36950287
Iteration 3: -0.4114037
Iteration 4: -0.44097826
Iteration 5: -0.46120283
Iteration 6: -0.4747295
Iteration 7: -0.4836406
Iteration 8: -0.48945206
Iteration 9: -0.493217
Iteration 10: -0.49564552
Iteration 11: -0.49720764
Iteration 12: -0.4982106
Iteration 13: -0.49885383
Iteration 14: -0.4992661
Iteration 15: -0.49953017
Iteration 16: -0.49969923
Iteration 17: -0.4998075
Iteration 18: -0.4998768
Iteration 19: -0.49992114
Iteration 20: -0.49994954
Iteration 21: -0.4999677
Iteration 22: -0.49997932
Iteration 23: -0.49998674
Iteration 24: -0.49999154
Iteration 25: -0.49999458
Iteration 26: -0.4999965
Iteration 27: -0.49999776
Iteration 28: -0.4999986
Iteration 29: -0.49999908
Iteration 30: -0.4999994
Iteration 31: -0.49999964
Iteration 32: -0.49999973
Iteration 33: -0.49999985
Iteration 34: -0.49999988
Iteration 35: -0.4999999
Iteration 36: -0.49999997
Iteration 37: -0.49999997
Iteration 38: -0.49999997
Iteration 39: -0.5
Minimum location: [fromList [] [0.23235208],fromList [] [-1.0176479]]
Minimum value: -0.5
```