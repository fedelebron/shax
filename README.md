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
      x2 :: f32[] = add x0 :: f32[] x1 :: f32[]
      x3 :: f32[] = sin x2 :: f32[]
      x4 :: f32[] = cos x2 :: f32[]
      x5 :: f32[] = mul x3 :: f32[] x4 :: f32[]
  in [x5 :: f32[]]
Iteration 1: -0.23971277
Iteration 2: -0.31265023
Iteration 3: -0.36950287
Iteration 4: -0.4114037
Iteration 5: -0.44097826
Iteration 6: -0.46120283
Iteration 7: -0.4747295
Iteration 8: -0.4836406
Iteration 9: -0.48945206
Iteration 10: -0.493217
Iteration 11: -0.49564552
Iteration 12: -0.49720764
Iteration 13: -0.4982106
Iteration 14: -0.49885383
Iteration 15: -0.4992661
Iteration 16: -0.49953017
Iteration 17: -0.49969923
Iteration 18: -0.4998075
Iteration 19: -0.4998768
Iteration 20: -0.49992114
Iteration 21: -0.49994954
Iteration 22: -0.4999677
Iteration 23: -0.49997932
Iteration 24: -0.49998674
Iteration 25: -0.49999154
Iteration 26: -0.49999458
Iteration 27: -0.4999965
Iteration 28: -0.49999776
Iteration 29: -0.4999986
Iteration 30: -0.49999908
Iteration 31: -0.4999994
Iteration 32: -0.49999964
Iteration 33: -0.49999973
Iteration 34: -0.49999985
Iteration 35: -0.49999988
Iteration 36: -0.4999999
Iteration 37: -0.49999997
Iteration 38: -0.49999997
Iteration 39: -0.49999997
Iteration 40: -0.5
Minimum location: [fromList [] [0.23235208],fromList [] [-1.0176479]]
Minimum value: -0.5
```

We can also see it perform tracing, type inference, linearization, and transposition:

```haskell
medium :: forall a. (HNP a, Ord a, Floating a) => a -> a -> a
medium x0 x8 =
        let x1 = broadcast [1] [2, 6] x0
            x2 = sin x1
            x3 = cos x2
            x4 = x2 + x3
            x5 = exp x4
            x6 = transpose [1, 0] x4
            x7 = x5 `dot` x6
            x9 = x7 * x8
         in x9
```

Here `HNP` acts as a standin for a NumPy-like API.

```
After tracing:
def medium(arg0 :: f32[6], 
           arg1 :: f32[2, 2]) =
  let x0 = arg0
      x1 = broadcast [1] [2,6] x0
      x2 = sin x1
      x3 = cos x2
      x4 = add x2 x3
      x5 = exp x4
      x6 = transpose [1,0] x4
      x7 = dot x5 x6
      x8 = arg1
      x9 = mul x7 x8
  in [x9]
After type inference:
def medium(arg0 :: f32[6], 
           arg1 :: f32[2, 2]) =
  let x0 :: f32[6] = arg0
      x1 :: f32[2, 6] = broadcast [1] [2,6] x0
      x2 :: f32[2, 6] = sin x1
      x3 :: f32[2, 6] = cos x2
      x4 :: f32[2, 6] = add x2 x3
      x5 :: f32[2, 6] = exp x4
      x6 :: f32[6, 2] = transpose [1,0] x4
      x7 :: f32[2, 2] = dot x5 x6
      x8 :: f32[2, 2] = arg1
      x9 :: f32[2, 2] = mul x7 x8
  in [x9]
Linearized definition(s):
def medium(arg0 :: f32[6], 
           arg1 :: f32[2, 2]) =
  let x0 :: f32[6] = arg0
      x1 :: f32[2, 6] = broadcast [1] [2,6] x0
      x2 :: f32[2, 6] = sin x1
      x3 :: f32[2, 6] = cos x1
      x4 :: f32[2, 6] = cos x2
      x5 :: f32[2, 6] = sin x2
      x6 :: f32[2, 6] = negate x5
      x7 :: f32[2, 6] = add x2 x4
      x8 :: f32[2, 6] = exp x7
      x9 :: f32[6, 2] = transpose [1,0] x7
      x10 :: f32[1, 2, 6] = reshape [1,2,6] x8
      x11 :: f32[1, 6, 2] = reshape [1,6,2] x9
      x12 :: f32[1, 2, 2] = dot_general [2] [1] [0] [0] x10 x11
      x13 :: f32[2, 2] = reshape [2,2] x12
      x14 :: f32[2, 2] = arg1
      x15 :: f32[2, 2] = mul x13 x14
  in [x15, x3, x6, x8, x10, x11, x13, x14]
def dmedium(arg0 :: f32[6], 
            arg1 :: f32[2, 2], 
            arg2 :: f32[2, 6], 
            arg3 :: f32[2, 6], 
            arg4 :: f32[2, 6], 
            arg5 :: f32[1, 2, 6], 
            arg6 :: f32[1, 6, 2], 
            arg7 :: f32[2, 2], 
            arg8 :: f32[2, 2]) =
  let x0 :: f32[6] = arg0
      x1 :: f32[2, 6] = broadcast [1] [2,6] x0
      x2 :: f32[2, 6] = arg2
      x3 :: f32[2, 6] = mul x2 x1
      x4 :: f32[2, 6] = arg3
      x5 :: f32[2, 6] = mul x4 x3
      x6 :: f32[2, 6] = add x3 x5
      x7 :: f32[2, 6] = arg4
      x8 :: f32[2, 6] = mul x7 x6
      x9 :: f32[6, 2] = transpose [1,0] x6
      x10 :: f32[1, 2, 6] = reshape [1,2,6] x8
      x11 :: f32[1, 6, 2] = reshape [1,6,2] x9
      x12 :: f32[1, 2, 6] = arg5
      x13 :: f32[1, 6, 2] = arg6
      x14 :: f32[1, 2, 2] = dot_general [2] [1] [0] [0] x12 x11
      x15 :: f32[1, 2, 2] = dot_general [2] [1] [0] [0] x10 x13
      x16 :: f32[1, 2, 2] = add x14 x15
      x17 :: f32[2, 2] = reshape [2,2] x16
      x18 :: f32[2, 2] = arg1
      x19 :: f32[2, 2] = arg7
      x20 :: f32[2, 2] = arg8
      x21 :: f32[2, 2] = mul x19 x18
      x22 :: f32[2, 2] = mul x20 x17
      x23 :: f32[2, 2] = add x21 x22
  in [x23]
f(x):
[┌───────────────────┐
 │18.365597 36.731194│
 │ 55.09679  73.46239│
 └───────────────────┘]
df(dx):
[┌───────────────────┐
 │22.402422 44.804844│
 │ 67.20727  89.60969│
 └───────────────────┘]
Transposed definition(s):
def medium(arg0 :: f32[6], 
           arg1 :: f32[2, 2]) =
  let x0 :: f32[6] = arg0
      x1 :: f32[2, 6] = broadcast [1] [2,6] x0
      x2 :: f32[2, 6] = sin x1
      x3 :: f32[2, 6] = cos x1
      x4 :: f32[2, 6] = cos x2
      x5 :: f32[2, 6] = sin x2
      x6 :: f32[2, 6] = negate x5
      x7 :: f32[2, 6] = add x2 x4
      x8 :: f32[2, 6] = exp x7
      x9 :: f32[6, 2] = transpose [1,0] x7
      x10 :: f32[1, 2, 6] = reshape [1,2,6] x8
      x11 :: f32[1, 6, 2] = reshape [1,6,2] x9
      x12 :: f32[1, 2, 2] = dot_general [2] [1] [0] [0] x10 x11
      x13 :: f32[2, 2] = reshape [2,2] x12
      x14 :: f32[2, 2] = arg1
      x15 :: f32[2, 2] = mul x13 x14
  in [x15, x3, x6, x8, x10, x11, x13, x14]
def dmediumt(arg0 :: f32[2, 2], 
             arg1 :: f32[2, 6], 
             arg2 :: f32[2, 6], 
             arg3 :: f32[2, 6], 
             arg4 :: f32[1, 2, 6], 
             arg5 :: f32[1, 6, 2], 
             arg6 :: f32[2, 2], 
             arg7 :: f32[2, 2]) =
  let x0 :: f32[2, 2] = arg0
      x1 :: f32[2, 6] = arg1
      x2 :: f32[2, 6] = arg2
      x3 :: f32[2, 6] = arg3
      x4 :: f32[1, 2, 6] = arg4
      x5 :: f32[1, 6, 2] = arg5
      x6 :: f32[2, 2] = arg6
      x7 :: f32[2, 2] = arg7
      x8 :: f32[2, 2] = mul x7 x0
      x9 :: f32[2, 2] = mul x6 x0
      x10 :: f32[1, 2, 2] = reshape [1,2,2] x8
      x11 :: f32[1, 2, 6] = transpose [0,2,1] x5
      x12 :: f32[1, 2, 6] = dot_general [2] [1] [0] [0] x10 x11
      x13 :: f32[1, 6, 2] = transpose [0,2,1] x4
      x14 :: f32[1, 6, 2] = dot_general [2] [1] [0] [0] x13 x10
      x15 :: f32[6, 2] = reshape [6,2] x14
      x16 :: f32[2, 6] = reshape [2,6] x12
      x17 :: f32[2, 6] = transpose [1,0] x15
      x18 :: f32[2, 6] = mul x3 x16
      x19 :: f32[2, 6] = add x18 x17
      x20 :: f32[2, 6] = mul x2 x19
      x21 :: f32[2, 6] = add x20 x19
      x22 :: f32[2, 6] = mul x1 x21
      x23 :: f32[6] = reduce_sum [0] x22
  in [x23, x9]
f(x) (again):
[┌───────────────────┐
 │18.365597 36.731194│
 │ 55.09679  73.46239│
 └───────────────────┘]
dft(ct):
[┌─────────────────────────────────────────────────────────────────┐
 │ 62.275585 -40.658405  -224.7738  -41.52993   8.645847  162.95436│
 └─────────────────────────────────────────────────────────────────┘,
 ┌───────────────────┐
 │36.731194  55.09679│
 │ 73.46239  91.82799│
 └───────────────────┘]
```