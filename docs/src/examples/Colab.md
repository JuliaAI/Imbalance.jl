# Google Colab

It is possible to run tutorials found in the examples section or API documentation on Google Colab (using provided link or icon). It should be evident how so by launching the notebook. This section describes what happens under the hood.

- The first cell runs the following bash script to install Julia:

```julia
%%capture
%%shell
if ! command -v julia 3>&1 > /dev/null
then
    wget -q 'https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.2-linux-x86_64.tar.gz' \
        -O /tmp/julia.tar.gz
    tar -x -f /tmp/julia.tar.gz -C /usr/local --strip-components 1
    rm /tmp/julia.tar.gz
fi
julia -e 'using Pkg; pkg"add IJulia; precompile;"'
echo 'Done'
```

- Once that is done, one can change the runtime to `Julia` by choosing `Runtime` from the toolbar then `Change runtime type` and at this point they can delete the cell

Sincere thanks to [Julia-on-Colab](https://github.com/Dsantra92/Julia-on-Colab) for making this possible.
