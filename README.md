# miopen
Go bindings for miopen (Radeon)

As of right now 10/29/2019. This works without modifications to the source code.  
This is the hip version of miopen.  

## Must install rocm,miopen and rocblas.  

First:
https://rocm.github.io/install.html

Then:
```
sudo apt-get install rocblas
sudo apt-get install miopengemm miopen-hip
```

## Packages Needed:
```
go get github.com/dereklstinson/cutil
go get github.com/dereklstinson/half
```

## Use a hip binding.

To allocate memory, streams, etc ...

```
go get github.com/dereklstinson/hip

```