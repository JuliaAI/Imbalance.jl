### class_counts
const ERR_MISSING_CLASS(c) = "Error: found class $c in y that is not in ratios."
const ERR_INVALID_RATIO(c) = "Error: ratio for class $c must be greater than 0."
const WRN_UNDERSAMPLE(new_ratio, label, less_counts, old_ratio) =
    "ratio $new_ratio for class $label implies that the class \
     should have $less_counts less samples because it is already $old_ratio \
     of the majority class but SMOTE cannot undersample.
     Will skip oversampling for this class."


### table_wrappers
const ERR_TABLE_TYPE(t) = "Error: expected a table or matrix but got a data of type $t."


