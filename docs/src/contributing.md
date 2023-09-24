# Directory Structure
The folder structure is as follows:
```
.
├── Imbalance.jl             # entry point to package
├── generic_resample.jl      # functions used in all resampling methods
├── generic_encoder.jl       # used in all resampling methods that deal with categorical data
├── table_wrappers.jl        # generalizes a function that operates on matrices to tables
├── class_counts.jl          # used to compute number of data points to add or remove
├── common                   # has julia files for common docs, error strings and utils
├── distance_metrics         # has distance metrics used by some resampling methods
├── oversampling_methods     # all oversampling methods live here
├── undersampling_methods    # all undersampling methods live here
└── extras.jl                # extra functions like generating data or checking balance
```

The purpose of each file is further documented therein at the beginning of the file. The files are ordered here in the recommended order of checking. 

Any method resampling method implemented in the `oversampling_methods` or `undersampling_methods` folder takes the following structure:
```
├── resample_method          # contains implementation and interfaces for a resampling method
│   ├── interface_mlj.jl     # implements MLJ interface for the method
│   ├── interface_tables.jl  # implements Tables.jl interface for the method
│   └── resample_method.jl   # implements the method itself (pure functional interface)
```

# Adding New Resampling Methods
- Make a new folder `resample_method` for the method in the oversampling_methods or undersampling_methods
- Implement in `resample_method/resample_method.jl` the method over matrices for one minority class
- Use `generic_oversample.jl` to generalize it to work on the whole data
- Use `table_wrapper.jl` to generalize the method to work on tables and possibly use `generic_encoder.jl`
- Implement the `MLJ` interface for the method in `resample_method/interface_mlj`
- Implement the `TableTransforms` interface for the method in `resample_method/interface_tables.jl`
- Use the rest of the files according to their description
- Testing and documentation should be done in parallel

# Adding New Tutorials
- Make a new notebook with the tutorial in the `examples` folder found in `docs/src/examples`
- Run the notebook so that the output is shown below each cell
- If the notebook produces visuals then save and load them in the notebook
- Convert it to markdown by using Python to run `from convert import convert_to_md; convert_to_md('<filename>')`
- Set a title, description, image and links for it in the dictionary found in `docs/examples.jl`
- For the colab link, you do not need to upload anything just follow the link pattern in the file