

"""
Given a categorical array of discrete labels and a dictionary of ratios, return a dictionary
of the number of extra samples needed for each class to achieve the given ratio 
relative to the majority class.

# Arguments
- `y::AbstractVector`: An abstract vector of class labels
$DOC_RATIOS_ARGUMENT

# Returns
- `Dict`: A dictionary mapping each class to the number of extra samples needed for
    that class to achieve the given ratio relative to the majority class.
"""
const ERR_MISSING_CLASS(c) = "Error: found class $c in y that is not in ratios."
const ERR_INVALID_RATIO(c) = "Error: ratio for class $c must be greater than 0."
const WRN_UNDERSAMPLE(new_ratio, label, less_counts, old_ratio) =
    "ratio $new_ratio for class $label implies that the class \
     should have $less_counts less samples because it is already $old_ratio \
     of the majority class but SMOTE cannot undersample.
     Will skip oversampling for this class."
# Method for handling ratios as a dictionary
function get_class_counts(y::AbstractVector, ratios::Dict{T,<:AbstractFloat}) where {T}
    label_counts = countmap(y)
    majority_count = maximum(values(label_counts))
    extra_counts = OrderedDict{T,Int}()

    # each class needs to be the size specified in `ratios`
    for (label, count) in label_counts
        (label in keys(ratios)) || throw(ERR_MISSING_CLASS(label))
        ratios[label] > 0 || throw(ERR_INVALID_RATIO(label))
        extra_counts[label] =
            calculate_extra_counts(ratios[label], majority_count, count, label)
    end
    return extra_counts
end

# Method for handling ratios as AbstractFloat
function get_class_counts(y::AbstractVector{T}, ratio::AbstractFloat) where {T}
    label_counts = countmap(y)
    majority_count = maximum(values(label_counts))
    extra_counts = OrderedDict{T,Int}()

    # each class needs to be the size specified in `ratio`
    for (label, count) in label_counts
        extra_counts[label] = calculate_extra_counts(ratio, majority_count, count, label)
    end

    return extra_counts
end

# Default method for when ratios is nothing
function get_class_counts(y::AbstractVector)
    get_class_counts(y, 1.0)
end
function get_class_counts(y::AbstractVector, ratio::Nothing)
    get_class_counts(y, 1.0)
end

"""
Helper function for calculating the number of extra samples needed for a class given a ratio, its count and majoiry count.
"""
function calculate_extra_counts(
    ratio::AbstractFloat,
    majority_count::Integer,
    label_count::Integer,
    label,
)
    extra_count = Int(round(ratio * majority_count)) - label_count
    if extra_count < 0
        old_ratio = label_count / majority_count
        new_ratio = ratio
        less_counts = extra_count
        @warn WRN_UNDERSAMPLE(new_ratio, label, less_counts, old_ratio)
        extra_count = 0
    end
    return extra_count
end
