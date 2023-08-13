# --- Transformers --- #
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import \
    StandardScaler, \
    RobustScaler, \
    OneHotEncoder
from resources.customtransformers import \
    DropConstantColumns, \
    DropDuplicateColumns, \
    AddNonZeroCount, \
    CustomSum, \
    CustomImputer, \
    AddNoneCount, \
    CustomEncoder, \
    PrefixScaler

# --- Pipeline Building --- #
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def build_prep() -> Pipeline:
    """
    Builds base pipeline.
    """
    prep = Pipeline(
        steps=[
            (
                "DropConstantColumns",
                DropConstantColumns(also=["ID"])
            ),
            (
                "DropDuplicateColumns",
                DropDuplicateColumns()
            ),
            (
                "NoneZeroCountSaldo",
                AddNonZeroCount(prefix="saldo")
            ),
            (
                "SumSaldo",
                CustomSum(prefix="saldo")
            ),
            (
                "NoneZeroCountImp",
                AddNonZeroCount(prefix="imp")
            ),
            (
                "SumImp",
                CustomSum(prefix="imp")
            ),
            (
                "ImputeNanDelta",
                CustomImputer(prefix="delta", to_replace=9999999999)
            ),
            (
                "NoneCountDelta",
                AddNoneCount(prefix="delta")
            ),
            (
                "NonZeroCountDelta",
                AddNonZeroCount(prefix="delta")
            ),
            (
                "SumDelta",
                CustomSum(prefix="delta")
            ),
            (
                "NonZeroContInd",
                AddNonZeroCount(prefix="ind")
            ),
            (
                "NonZeroCountNum",
                AddNonZeroCount(prefix="num")
            ),
            (
                "SumNum",
                CustomSum(prefix="num")
            ),
            (
                "ImputeNanVar3",
                CustomImputer(prefix="var3", to_replace=-999999)
            ),
            (
                "CustomEncoderVar36",
                CustomEncoder(colname="var36")
            ),
            (
                "CustomEncoderVar21",
                CustomEncoder(colname="var21")
            )
        ]
    )
    return prep

def build_prep_nan() -> Pipeline:
    """
    Builds base pipeline with nan imputation.
    """
    prep_nan = Pipeline(
        steps=[
            ("prep", build_prep()),
            ("NoneCountVar3", AddNoneCount(prefix="var3")),
            ("nan", SimpleImputer(strategy="median"))
        ]
    )
    return prep_nan

def build_prep_cluster(n_comp=None):
    cat_cols = ["var36", "var21"]

    rbs_prefixes = [
        "saldo",
        "non_zero_count_saldo",
        "sum_of_saldo",
        "imp",
        "non_zero_count_imp",
        "sum_of_imp",
        "delta",
        "none_count_delta",
        "sum_of_delta",
        "non_zero_count_delta",
        "non_zero_count_ind",
        "num"
        "non_zero_count_num",
        "sum_of_num",
        "var3",
        "var15",
    ]

    ss_prefixes = [
        "var38"
    ]

    cat_tf = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(min_frequency=100, sparse_output=False)),
            ("ss", StandardScaler())
        ]
    )

    prep = Pipeline(
        steps=[
            ("base", build_prep()[:-2]),
            ("rbs", PrefixScaler(rbs_prefixes, RobustScaler())),
            ("ss", PrefixScaler(ss_prefixes, StandardScaler())),
            ("cat",ColumnTransformer([("ohe", cat_tf, cat_cols)], remainder='passthrough')),
            ("knn", KNNImputer(n_neighbors=5)),
            ("pca", PCA(n_components=n_comp))
        ]
    )

    return prep