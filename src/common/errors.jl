"""
This file contains any warning or error used in any function in this package.
"""


### class_counts
const ERR_INVALID_RATIO(c) = "ratio for class $c must be greater than 0."
const WRN_UNDERSAMPLE(new_ratio, label, less_counts, old_ratio) =
	"ratio $new_ratio for class $label implies that the class" *
	"should have $less_counts less samples because it is already $old_ratio" *
	"of the majority class but this algorithm cannot undersample." *
	"Will skip oversampling for this class."

const WRN_OVERSAMPLE(new_ratio, label, extra_counts, old_ratio) =
	"ratio $new_ratio for class $label implies that the class" *
	"should have $extra_counts more samples because it is already $old_ratio" *
	"of the minority class but this algorithm cannot oversample." *
	"Will skip undersampling for this class."


### table_wrappers
const ERR_TABLE_TYPE(t) = "Expected a table or matrix of appropriate element types but got a data of type $t."

### ROSE
const ERR_NEG_S(s) = "s must be >= 0 but got s=$s."

### SMOTE
const WRN_SINGLE_OBS = "class with a single point will be ignored; point has no neighbors"
const ERR_NONPOS_K(k) = "Number of nearest neighbors `k` must be greater than 0 but got $k."
const WRN_K_TOO_BIG(k, n) =
    "k=$k is  larger than or equal to the number of samples for the data which is ($n). Will set k=$n-1"

### BorderlineSMOTE1
const ERR_NONPOS_M(m) = " Number of nearest neighbors `m` must be greater than 0 but got $m."
const WRN_M_TOO_BIG(m, n) =
	"m=$m is larger than or equal to the number of samples for the data which is ($n). Will set m=$m-1"
const ERR_NO_BORDERLINE = " No borderline points were found and thus, data cannot be oversampled with this method."
const WRN_NO_BORDERLINE_CLASS = "Cannot oversample a class with no borderline points. Skipping."
const INFO_BORDERLINE_PTS(y1_stats) = "After filtering, the mapping from each class to number of borderline points is ($y1_stats)."


### SMOTENC
const ERR_BAD_MIXED_COL_TYPES(bad_cols, bad_types) =
	"Columns at indices $(bad_cols) are neither categorical nor continuous.\n Categorical columns must have scitype Multiclass or OrderedFactor and continuous columns must have scitype Count or Continuous.\n However, said columns have scitypes $(bad_types)."
# May need to do similar checks for matrices but they are just numbers
const ERR_WRNG_TREE(knn_tree) =
	"Expected KNN tree to be either \"Brute\" or \"Ball\" but given is \"$knn_tree\""

### SMOTEN and RandomWalk
const ERR_BAD_NOM_COL_TYPES(bad_cols, bad_types) =
    "Columns at indices $(bad_cols) are not categorical.\n Categorical columns must have scitype Multiclass or OrderedFactor.\n However, said columns have scitypes $(bad_types)."


### Cluster Undersampling
const ERR_INVALID_MODE = "mode must be either \"center\" or \"nearest\" "

### ENN Undersampling
const ERR_KEEP_CONDS = "keep_condition must be one of: \"exists\", \"mode\", \"only mode\", \"all\""

