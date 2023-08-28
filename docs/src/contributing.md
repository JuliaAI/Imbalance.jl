# Directory Structure
The folder structure is as follows:
```
.
├── Imbalance.jl             # entry point to package
├── generic_oversample.jl    # used in all resampling methods
├── generic_encoder.jl       # used in all resampling methods that deal with categorical data
├── table_wrappers.jl        # generalizes a function that operates on matrices to tables
├── class_counts.jl          # used to compute number of data points to add
├── resample_method          # contains implementation and interface for a resampling method
│   ├── interface_mlj.jl     # implements MLJ interface for the method
│   ├── interface_tables.jl  # implements Tables.jl interface for the method
│   └── resample_method.jl   # implements the method itself
├── common_smote.jl          # implementation functions shared by different variants of SMOTE
├── commondocs.jl            # documentation that is common over many functions
├── errors.jl                # any error or warning used in the package
├── utils.jl                 # utility functions 
└── mlj_interface.jl         # registers metadata for the MLJ interface of each method
```

The purpose of each file is further documented therein at the beginning of the file. The files are ordered here in the recommended order of checking.


# Adding New Resampling Methods
- Make a new folder `resample_method` for the method
- Implement in `resample_method/resample_method.jl` the method over matrices for one minority class (else skip next step if that's not possible) and document implemented functions
- Use `generic_oversample.jl` to generalize it to work on the whole data
- Use `table_wrapper.jl` to generalize the method to work on tables and possibly `generic_encoder.jl`
- Make a test file `test/resample_method.jl` and test the implemented functions
- Implement the `MLJ` interface for the method in `resample_method/interface_mlj`
- Implement the `TableTransforms` interface for the method in `resample_method/interface_tables.jl`
- Use the rest of the files according to their description

# Adding New Tutorials
- Make a new notebook with the tutorial in the `examples` folder
- Run the notebook so that the output is shown below each cell
- If the notebook produces visuals then save and load them in the notebook
- Run the Python script found in the last cell of other tutorials. Now it's in the docs folder in markdown
- Set a title, description, image and links for it in the dictionary found in `docs/examples.jl`
- For the colab link, you do not need to upload anything just follow the link pattern in the file