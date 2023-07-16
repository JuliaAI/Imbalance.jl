
function get_class_counts(y, ratios=nothing)
    label_counts = group_counts(y)
    majority_count = maximum(values(label_counts))
    extra_counts = Dict()
    if isnothing(ratios)
        # each class needs to be the same size as the majority class
        for (label, count) in label_counts
            # make a new key in the dictionary called class and set it to the majority count - the count of the class
            extra_counts[label] = majority_count - count
        end
    else
        # each class needs to be the size specified in `ratios`
        for (label, _) in label_counts
            if !(label in keys(ratios))
                msg = "ratios must contain a key for each class in y."
                error(ArgumentError(msg))
            elseif ratios[label] < 0
                    msg = "ratio for any class must be greater than or equal to 0."
                    error(ArgumentError(msg))
            else
                extra_counts[label] = Int(round(ratios[label] * majority_count)) - label_counts[label]
                if extra_counts[label] < 0
                    old_ratio = label_counts[label] / majority_count
                    new_ratio = ratios[label]
                    less_counts = extra_counts[label] 
                    # produce warning
                    @warn "ratio $new_ratio for class $label implies that the class should have $less_counts less samples because it is already $old_ratio of the majority class but SMOTE cannot undersample.\n Will skip oversampling for this class."
                    extra_counts[label] = 0
                end
            end
        end
    end
    return extra_counts
end





function check_is_table(data)
    if !Tables.istable(data)
        T = typeof(data)
        msg = "Expected tabular data, matrix or vector.Got data of type $T."
        error(ArgumentError(msg))
    end
end

randobs(rng::AbstractRNG, X) = getobs(X, rand(rng, 1:numobs(X)))  
randobs(rng::AbstractRNG, X, n) = getobs(X, rand(rng, 1:numobs(X), n))
