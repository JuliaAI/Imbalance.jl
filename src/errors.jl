### class_counts
const ERR_MISSING_CLASS(c) = "Error: found class $c in y that is not in ratios."
const ERR_INVALID_RATIO(c) = "Error: ratio for class $c must be greater than 0."
const WRN_UNDERSAMPLE(new_ratio, label, less_counts, old_ratio) =
    "Warning: ratio $new_ratio for class $label implies that the class \
     should have $less_counts less samples because it is already $old_ratio \
     of the majority class but SMOTE cannot undersample.
     Will skip oversampling for this class."


### table_wrappers
const ERR_TABLE_TYPE(t) = "Error: expected a table or matrix but got a data of type $t."

### ROSE
const ERR_NEG_S(s) = "Error: s must be >= 0 but got $s."

### SMOTE
const WRN_SINGLE_OBS = "Warning: class with a single will be ignored; point has no neighbors"
const ERR_NONPOS_K(k) = "Error: k must be greater than 0 but got $k."
const WRN_K_TOO_BIG(k, n) =
    "Warning: k=$k is larger than the number of samples for the class which is ($n). Will set k=$n-1"

### SMOTENC
const ERR_BAD_MIXED_COL_TYPES(bad_cols, bad_types) =
    "Columns at indices $(bad_cols) are neither categorical nor continuous.\n Categorical columns must have scitype Multiclass or OrderedFactor and continuous columns must have scitype Count or Continuous.\n However, said columns have scitypes $(bad_types)."
# May need to do similar checks for matrices but they are just numbers

### SMOTEN
const ERR_BAD_NOM_COL_TYPES(bad_cols, bad_types) =
    "Columns at indices $(bad_cols) are not categorical.\n Categorical columns must have scitype Multiclass or OrderedFactor.\n However, said columns have scitypes $(bad_types)."
