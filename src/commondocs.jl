
const DOC_RATIOS_ARGUMENT = """
                            - `ratios=1.0`: A parameter that controls the amount of oversampling to be done for each class
                                - Can be a float and in this case each class will be oversampled to the size of the majority class times the float. By \
                                default, all classes are oversampled to the size of the majority class
                                - Can be a dictionary mapping each class to the ratio of the needed number of observations for that class to the \
                                initial number of observations of the majority class
                            """

const DOC_RNG_ARGUMENT = """
                         - `rng::Union{AbstractRNG, Integer}`: Either an `AbstractRNG` object or an `Integer` 
                             seed to be used with `Xoshiro`\n
                         """

const DOC_COMMON_K = """
                    - `k::Int=5`: Number of nearest neighbors to consider in the SMOTE algorithm. Should be within the range \
                        `0 < k < n` where n is the number of observations in the smallest class. It will be automatically set to \
                        `n-1` for any class where `n ≤ k`.
                     """

const DOC_TRY_PERSERVE_ARGUMENT = """
                                 - `try_preserve_type::Bool=true`: Defaults to true and means that the function will try to perserve the type of the 
                                        table (e.g., `DataFrame`). However, for some tables this may not succeed and in this case the table returned will
                                        be a `Columntable` (named-tuple of vectors). 
                                 """

const DOC_COMMON_INPUTS = """
                          - `X`: A matrix or table of floats where each row is an observation from the dataset 

                          - `y`: An abstract vector of labels (e.g., strings) that correspond to the observations in `X`
                          """

const DOC_COMMON_OUTPUTS = """
                           - `Xover`: A matrix or table that includes original data and the new observations 
                                due to oversampling. depending on whether the input `X` is a matrix or table respectively

                           - `yover`: An abstract vector of labels that includes the original
                               labels and the new instances of them due to oversampling
                           """
