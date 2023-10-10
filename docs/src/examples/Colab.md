# Google Colab

It is possible to run tutorials found in the examples section on Google Colab. The same applies to the illustrative examples in the API documentation.
- Click the Colab icon link (shows up upon hovering if on the example if not visible)
- Paste and run the following in the first cell to install Julia

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

- Change the runtime to `Julia` by choosing `Runtime` from the toolbar then `Change runtime type` 
- `Pkg.add` Imbalance and any needed packages (those being used). For a comprehensive list:
```julia
Pkg.add(["Random", "CSV", "DataFrames", 
         "MLJ", "MLJBase", "Imbalance", "MLJBalancing", 
         "ScientificTypes", "CategoricalArrays", "Impute", 
         "TableTransforms",  "StatsBase",  
         "Plots", "Colors"
         ])
```
Sincere thanks to [Julia-on-Colab](https://github.com/Dsantra92/Julia-on-Colab) for making this possible
