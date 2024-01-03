# shax

This is a simple implementation of the tracing + linearization + transposition approach to automatic differentiation, as explained in the [You Only Linearize Once paper](https://arxiv.org/abs/2204.10923). One difference is we don't use explicit `dup` and `drop` nodes, since we're not creating a cost model. We also don't explicitly create the `unzip` transform, it's implicit in linearization.

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
            x10 = x7 + x8
            x11 = min x9 x10
            x12 = max x9 x10
            x13 = x12 - x11
            x14 = reduceSum [0] x13
         in x14
```

Here `HNP` acts as a standin for a NumPy-like API, providing `broadcast`, `transpose`, `dot`, and `reduceSum`.

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
      x10 = add x7 x8
      x11 = max x9 x10
      x12 = min x9 x10
      x13 = sub x11 x12
      x14 = reduce_sum [0] x13
  in [x14]
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
      x10 :: f32[2, 2] = add x7 x8
      x11 :: f32[2, 2] = max x9 x10
      x12 :: f32[2, 2] = min x9 x10
      x13 :: f32[2, 2] = sub x11 x12
      x14 :: f32[2] = reduce_sum [0] x13
  in [x14]
After rewrites (canonicalizing `dot`s and lowering `reduce_sum` to pointwise `+` and `slice`):
def medium(arg0 :: f32[6], 
           arg1 :: f32[2, 2]) =
  let x0 :: f32[6] = arg0
      x1 :: f32[2, 6] = broadcast [1] [2,6] x0
      x2 :: f32[2, 6] = sin x1
      x3 :: f32[2, 6] = cos x2
      x4 :: f32[2, 6] = add x2 x3
      x5 :: f32[2, 6] = exp x4
      x6 :: f32[6, 2] = transpose [1,0] x4
      x7 :: f32[1, 2, 6] = reshape [1,2,6] x5
      x8 :: f32[1, 6, 2] = reshape [1,6,2] x6
      x9 :: f32[1, 2, 2] = dot_general [2] [1] [0] [0] x7 x8
      x10 :: f32[2, 2] = reshape [2,2] x9
      x11 :: f32[2, 2] = arg1
      x12 :: f32[2, 2] = mul x10 x11
      x13 :: f32[2, 2] = add x10 x11
      x14 :: f32[2, 2] = max x12 x13
      x15 :: f32[2, 2] = min x12 x13
      x16 :: f32[2, 2] = sub x14 x15
      x17 :: f32[1, 2] = slice [0,0] [1,2] x16
      x18 :: f32[1, 2] = slice [1,0] [2,2] x16
      x19 :: f32[1, 2] = add x17 x18
      x20 :: f32[2] = reshape [2] x19
  in [x20]
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
      x16 :: f32[2, 2] = add x13 x14
      x17 :: f32[2, 2] = max x15 x16
      x18 :: f32[] = fromList [] [0.0]
      x19 :: f32[] = fromList [] [1.0]
      x20 :: f32[] = fromList [] [2.0]
      x21 :: f32[2, 2] = broadcast [] [2,2] x18
      x22 :: f32[2, 2] = broadcast [] [2,2] x19
      x23 :: f32[2, 2] = broadcast [] [2,2] x20
      x24 :: bool[2, 2] = eq x17 x15
      x25 :: f32[2, 2] = select x24 x22 x21
      x26 :: bool[2, 2] = eq x17 x16
      x27 :: f32[2, 2] = select x26 x23 x22
      x28 :: f32[2, 2] = div x25 x27
      x29 :: f32[2, 2] = sub x22 x28
      x30 :: f32[2, 2] = min x15 x16
      x31 :: f32[] = fromList [] [0.0]
      x32 :: f32[] = fromList [] [1.0]
      x33 :: f32[] = fromList [] [2.0]
      x34 :: f32[2, 2] = broadcast [] [2,2] x31
      x35 :: f32[2, 2] = broadcast [] [2,2] x32
      x36 :: f32[2, 2] = broadcast [] [2,2] x33
      x37 :: bool[2, 2] = eq x30 x15
      x38 :: f32[2, 2] = select x37 x35 x34
      x39 :: bool[2, 2] = eq x30 x16
      x40 :: f32[2, 2] = select x39 x36 x35
      x41 :: f32[2, 2] = div x38 x40
      x42 :: f32[2, 2] = sub x35 x41
      x43 :: f32[2, 2] = sub x17 x30
      x44 :: f32[1, 2] = slice [0,0] [1,2] x43
      x45 :: f32[1, 2] = slice [1,0] [2,2] x43
      x46 :: f32[1, 2] = add x44 x45
      x47 :: f32[2] = reshape [2] x46
  in [x47, x3, x6, x8, x10, x11, x13, x14, x28, x29, x41, x42]
def dmedium(arg0 :: f32[6], 
            arg1 :: f32[2, 2], 
            arg2 :: f32[2, 6], 
            arg3 :: f32[2, 6], 
            arg4 :: f32[2, 6], 
            arg5 :: f32[1, 2, 6], 
            arg6 :: f32[1, 6, 2], 
            arg7 :: f32[2, 2], 
            arg8 :: f32[2, 2], 
            arg9 :: f32[2, 2], 
            arg10 :: f32[2, 2], 
            arg11 :: f32[2, 2], 
            arg12 :: f32[2, 2]) =
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
      x24 :: f32[2, 2] = add x17 x18
      x25 :: f32[2, 2] = arg9
      x26 :: f32[2, 2] = arg10
      x27 :: f32[2, 2] = mul x25 x23
      x28 :: f32[2, 2] = mul x26 x24
      x29 :: f32[2, 2] = add x27 x28
      x30 :: f32[2, 2] = arg11
      x31 :: f32[2, 2] = arg12
      x32 :: f32[2, 2] = mul x30 x23
      x33 :: f32[2, 2] = mul x31 x24
      x34 :: f32[2, 2] = add x32 x33
      x35 :: f32[2, 2] = sub x29 x34
      x36 :: f32[1, 2] = slice [0,0] [1,2] x35
      x37 :: f32[1, 2] = slice [1,0] [2,2] x35
      x38 :: f32[1, 2] = add x36 x37
      x39 :: f32[2] = reshape [2] x38
  in [x39]
f(x):
[┌───────────────────┐
 │34.731194  67.46239│
 └───────────────────┘]
df(dx):
[┌───────────────────┐
 │42.804844 120.34088│
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
      x16 :: f32[2, 2] = add x13 x14
      x17 :: f32[2, 2] = max x15 x16
      x18 :: f32[] = fromList [] [0.0]
      x19 :: f32[] = fromList [] [1.0]
      x20 :: f32[] = fromList [] [2.0]
      x21 :: f32[2, 2] = broadcast [] [2,2] x18
      x22 :: f32[2, 2] = broadcast [] [2,2] x19
      x23 :: f32[2, 2] = broadcast [] [2,2] x20
      x24 :: bool[2, 2] = eq x17 x15
      x25 :: f32[2, 2] = select x24 x22 x21
      x26 :: bool[2, 2] = eq x17 x16
      x27 :: f32[2, 2] = select x26 x23 x22
      x28 :: f32[2, 2] = div x25 x27
      x29 :: f32[2, 2] = sub x22 x28
      x30 :: f32[2, 2] = min x15 x16
      x31 :: f32[] = fromList [] [0.0]
      x32 :: f32[] = fromList [] [1.0]
      x33 :: f32[] = fromList [] [2.0]
      x34 :: f32[2, 2] = broadcast [] [2,2] x31
      x35 :: f32[2, 2] = broadcast [] [2,2] x32
      x36 :: f32[2, 2] = broadcast [] [2,2] x33
      x37 :: bool[2, 2] = eq x30 x15
      x38 :: f32[2, 2] = select x37 x35 x34
      x39 :: bool[2, 2] = eq x30 x16
      x40 :: f32[2, 2] = select x39 x36 x35
      x41 :: f32[2, 2] = div x38 x40
      x42 :: f32[2, 2] = sub x35 x41
      x43 :: f32[2, 2] = sub x17 x30
      x44 :: f32[1, 2] = slice [0,0] [1,2] x43
      x45 :: f32[1, 2] = slice [1,0] [2,2] x43
      x46 :: f32[1, 2] = add x44 x45
      x47 :: f32[2] = reshape [2] x46
  in [x47, x3, x6, x8, x10, x11, x13, x14, x28, x29, x41, x42]
def dmediumt(arg0 :: f32[2], 
             arg1 :: f32[2, 6], 
             arg2 :: f32[2, 6], 
             arg3 :: f32[2, 6], 
             arg4 :: f32[1, 2, 6], 
             arg5 :: f32[1, 6, 2], 
             arg6 :: f32[2, 2], 
             arg7 :: f32[2, 2], 
             arg8 :: f32[2, 2], 
             arg9 :: f32[2, 2], 
             arg10 :: f32[2, 2], 
             arg11 :: f32[2, 2]) =
  let x0 :: f32[2] = arg0
      x1 :: f32[2, 6] = arg1
      x2 :: f32[2, 6] = arg2
      x3 :: f32[2, 6] = arg3
      x4 :: f32[1, 2, 6] = arg4
      x5 :: f32[1, 6, 2] = arg5
      x6 :: f32[2, 2] = arg6
      x7 :: f32[2, 2] = arg7
      x8 :: f32[2, 2] = arg8
      x9 :: f32[2, 2] = arg9
      x10 :: f32[2, 2] = arg10
      x11 :: f32[2, 2] = arg11
      x12 :: f32[1, 2] = reshape [1,2] x0
      x13 :: f32[2, 2] = pad [(1,0),(0,0)] 0.0 x12
      x14 :: f32[2, 2] = pad [(0,1),(0,0)] 0.0 x12
      x15 :: f32[2, 2] = add x14 x13
      x16 :: f32[2, 2] = negate x15
      x17 :: f32[2, 2] = mul x11 x16
      x18 :: f32[2, 2] = mul x10 x16
      x19 :: f32[2, 2] = mul x9 x15
      x20 :: f32[2, 2] = mul x8 x15
      x21 :: f32[2, 2] = add x19 x17
      x22 :: f32[2, 2] = add x20 x18
      x23 :: f32[2, 2] = mul x7 x22
      x24 :: f32[2, 2] = mul x6 x22
      x25 :: f32[2, 2] = add x24 x21
      x26 :: f32[2, 2] = add x23 x21
      x27 :: f32[1, 2, 2] = reshape [1,2,2] x26
      x28 :: f32[1, 2, 6] = transpose [0,2,1] x5
      x29 :: f32[1, 2, 6] = dot_general [2] [1] [0] [0] x27 x28
      x30 :: f32[1, 6, 2] = transpose [0,2,1] x4
      x31 :: f32[1, 6, 2] = dot_general [2] [1] [0] [0] x30 x27
      x32 :: f32[6, 2] = reshape [6,2] x31
      x33 :: f32[2, 6] = reshape [2,6] x29
      x34 :: f32[2, 6] = transpose [1,0] x32
      x35 :: f32[2, 6] = mul x3 x33
      x36 :: f32[2, 6] = add x35 x34
      x37 :: f32[2, 6] = mul x2 x36
      x38 :: f32[2, 6] = add x37 x36
      x39 :: f32[2, 6] = mul x1 x38
      x40 :: f32[6] = reduce_sum [0] x39
  in [x40, x25]
f(x) (again):
[┌───────────────────┐
 │34.731194  67.46239│
 └───────────────────┘]
dft(ct):
[┌─────────────────────────────────────────────────────────────────┐
 │  71.61691 -46.757164 -258.48987 -47.759422   9.942725  187.39752│
 └─────────────────────────────────────────────────────────────────┘,
 ┌───────────────────┐
 │-86.82799 156.29037│
 │ 86.82799 156.29037│
 └───────────────────┘]
```