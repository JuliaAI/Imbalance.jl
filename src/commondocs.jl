
const DOC_RATIOS_ARGUMENT = """
                            - `ratios=nothing`: A parameter that controls the amount of oversampling to be done for each class.
                                - Can be a dictionary mapping each class to the ratio of the needed number of observations for that 
                                class to the initial number of observations of the majority class.
                                - Can be nothing and in this case each class will be oversampled to the size of the majority class.
                                - Can be a float and in this case each class will be oversampled to the size of the majority class times the float.
                            """

const DOC_RNG_ARGUMENT = """
                         - `rng::Union{AbstractRNG, Integer}`: Either an `AbstractRNG` object or an `Integer` 
                             seed to be used with `StableRNG`.\n
                         """


const DOC_COMMON_INPUTS = """
                          - `X`: A matrix or table where each row is an observation (vector) of floats

                          - `y`: An abstract vector of labels that correspond to the observations in `X`
                          """

const DOC_COMMON_OUTPUTS = """
                           - `Xover`: A matrix or table like X (if possible, else a columntable) depending on whether X 
                               is a matrix or table respectively that includes original data and the new observations 
                               due to oversampling.

                           - `yover`: An abstract vector of labels that includes the original
                               labels and the new instances of them due to oversampling.
                           """
