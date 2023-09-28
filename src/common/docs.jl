"""
This file contains some docs that are used by multiple oversampling methods.
"""


const COMMON_DOCS = Dict(
    "INPUTS" => """
    - `X`: A matrix or table of floats where each row is an observation from the dataset 

    - `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`
    """,

    "OUTPUTS" => """
    - `Xover`: A matrix or table that includes original data and the new observations 
        due to oversampling. depending on whether the input `X` is a matrix or table respectively

    - `yover`: An abstract vector of labels corresponding to `Xover`
    """,

    "OUTPUTS-UNDER" => """
    - `X_under`: A matrix or table that includes the data after undersampling 
        depending on whether the input `X` is a matrix or table respectively

    - `y_under`: An abstract vector of labels corresponding to `X_under`
    """,
    
    "RATIOS" => """
    - `ratios=1.0`: A parameter that controls the amount of oversampling to be done for each class
        - Can be a float and in this case each class will be oversampled to the size of the majority class times the float. By """*
        """default, all classes are oversampled to the size of the majority class
        - Can be a dictionary mapping each class label to the float ratio for that class
    """,
    "RATIOS-UNDERSAMPLE" => """
    - `ratios=1.0`: A parameter that controls the amount of undersampling to be done for each class
        - Can be a float and in this case each class will be undersampled to the size of the minority class times the float. By """*
        """default, all classes are undersampled to the size of the minority class
        - Can be a dictionary mapping each class label to the float ratio for that class
    """,
    "MIN-RATIOS-UNDERSAMPLE" => """
    - `min_ratios=1.0`: A parameter that controls the maximum amount of undersampling to be done for each class. If this algorithm
        cleans the data to an extent that this is violated, some of the cleaned points will be revived randomly so that it is satisfied.
        - Can be a float and in this case each class will be at most undersampled to the size of the minority class times the float. By """*
        """default, all classes are undersampled to the size of the minority class
        - Can be a dictionary mapping each class label to the float minimum ratio for that class
    """,
    "FORCE-MIN-RATIOS" => """
    - `force_min_ratios=false`: If `true`, and this algorithm cleans the data such that the ratios for each class
        exceed those specified in `min_ratios` then further undersampling will be perform so that the final ratios
        are equal to `min_ratios`.
    """,
    "RNG" => """
    - `rng::Union{AbstractRNG, Integer}=default_rng()`: Either an `AbstractRNG` object or an `Integer` 
        seed to be used with `Xoshiro`
    """,

    "K" => """
    - `k::Integer=5`: Number of nearest neighbors to consider in the algorithm. Should be within the range """*
        """`0 < k < n` where n is the number of observations in the smallest class. It will be automatically set to """*
        """`n-1` for any class where `n ≤ k`.
    """,
    "K-FULL" => """
    - `k::Integer=5`: Number of nearest neighbors to consider in the algorithm. Should be within the range """*
        """`0 < k < n` where n is the number of observations in the data. It will be automatically set to """*
        """`n-1` if `n ≤ k`.
    """,
    "TRY_PERSERVE_TYPE" => """
    - `try_preserve_type::Bool=true`: When `true`, the function will try to preserve the type of the input 
        table (e.g., `DataFrame`). However, for some tables this may not succeed and in this case the table returned will
        be a column table (named-tuple of vectors). This parameter is ignored if the input is a matrix.
    """,

)
