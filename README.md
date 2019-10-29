# miopen
Go bindings for miopen (Radeon)


It has been a train wreck trying to get this to work with gcc.  
Currently there is a bug with gcc and hip. 
I had to do some slight modifications to hip and miopen to get this to work with gcc.

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
```
go get github.com/dereklstinson/hip

```