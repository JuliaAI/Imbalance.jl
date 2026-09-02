"""
This file has the `get_class_counts` method which is used by `generic_oversample` and
hence, all the oversampling methods. It computes the number of points to oversample for
each class.
"""

"""

Given a categorical array of discrete labels and a dictionary of ratios, return a
dictionary with the number of extra samples needed for each class included in the input
dictionary to achieve the given ratio relative to the majority class.

# Arguments
- `y::AbstractVector`: An abstract vector of class labels
$(COMMON_DOCS["RATIOS"])

# Returns
- `extra_counts`: A dictionary mapping each class to the number of extra samples needed for
  that class to achieve the given ratio relative to the majority class.

"""
# Method for handling ratios as a dictionary
function get_class_counts(
    y::AbstractVector{T},
    ratios::AbstractDict;
    reference = "majority",
) where {T}
    label_counts = countmap(y)
    (reference in ["majority", "minority"]) || throw(())
    ref_count =
        (reference == "majority") ? maximum(values(label_counts)) :
        minimum(values(label_counts))
    # extra_counts shall map each class to the number of extra samples needed
    # for the class to satisfy ratios
    counts = OrderedDict{T, Int}()

    # each class needs to be the size specified in `ratios`
    for (label, count) in label_counts
        (label in keys(ratios)) || continue
        ratios[label] > 0 || throw((ERR_INVALID_RATIO(label)))
        counts[label] =
            (reference == "majority") ?
            calculate_extra_counts(ratios[label], ref_count, count, label) :
            calculate_undersampled_counts(ratios[label], ref_count, count, label)
    end
    return counts
end

# Method for handling ratios as AbstractFloat (dispaches the above on appropriate dict):
function get_class_counts(y::AbstractVector, ratio::AbstractFloat; reference = "majority")
    count_given_label = countmap(y)
    reference_count =
        reference == "majority" ? maximum(values(count_given_label)) :
        minimum(values(count_given_label))
    labels = keys(count_given_label) |> collect
    labeled_ratios = map(labels) do label
        count = count_given_label[label]
        return label => (count == reference_count ? 1.0 : ratio)
    end
    ratio_given_label = OrderedDict(labeled_ratios...)
    return get_class_counts(y, ratio_given_label; reference)
end

"""
Helper function for calculating the number of extra samples needed for a class given a
ratio, its count and majority count.  It assumes that if ratio is `a` and the majority
count is `N` then the class should have `aN` samples so it computes the number of extra
samples.
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

"""
Helper function for calculating the number of final samples needed for a class given a
ratio, its count and minority count.  It assumes that if ratio is `a` and the minority
count is `N` then the class should have `aN` samples.
"""
function calculate_undersampled_counts(
    ratio::AbstractFloat,
    minority_count::Integer,
    label_count::Integer,
    label,
)
    undersampled_count = Int(round(ratio * minority_count))
    if undersampled_count > label_count
        old_ratio = label_count / minority_count
        new_ratio = ratio
        extra_counts = undersampled_count - label_count
        @warn WRN_OVERSAMPLE(new_ratio, label, extra_counts, old_ratio)
        undersampled_count = label_count
    end
    return undersampled_count
end
