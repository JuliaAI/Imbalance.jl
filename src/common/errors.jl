"""
This file contains any warning or error used in any function in this package.
"""


### class_counts
const ERR_INVALID_RATIO(c) = "Error: ratio for class $c must be greater than 0."
const WRN_UNDERSAMPLE(new_ratio, label, less_counts, old_ratio) =
	"Warning: ratio $new_ratio for class $label implies that the class" *
	"should have $less_counts less samples because it is already $old_ratio" *
	"of the majority class but this algorithm cannot undersample." *
	"Will skip oversampling for this class."

const WRN_OVERSAMPLE(new_ratio, label, extra_counts, old_ratio) =
	"ratio $new_ratio for class $label implies that the class" *
	"should have $extra_counts more samples because it is already $old_ratio" *
	"of the minority class but this algorithm cannot oversample." *
	"Will skip undersampling for this class."


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
const ERR_WRNG_TREE(knn_tree) =
	"Expected KNN tree to be either 'BruteTree' or 'BallTree' but given is '$knn_tree'"

### SMOTEN
const ERR_BAD_NOM_COL_TYPES(bad_cols, bad_types) =
	"Columns at indices $(bad_cols) are not categorical.\n Categorical columns must have scitype Multiclass or OrderedFactor.\n However, said columns have scitypes $(bad_types)."
