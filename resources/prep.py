# --- Transformers --- #
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import \
    StandardScaler, \
    OneHotEncoder
from resources.customtransformers import \
    DropConstantColumns, \
    DropDuplicateColumns, \
    AddNonZeroCount, \
    CustomSum, \
    CustomImputer, \
    AddNoneCount, \
    CustomEncoder, \
    CustomLog

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

def build_prep_2() -> Pipeline:
    prep = Pipeline(
        steps=[
            ("prep", build_prep()),
            ("NoneCountVar3", AddNoneCount(prefix="var3")),
            ("drop_almost", DropConstantColumns(thresh=.99, ignore_prefix=["ind"])),
            ("nan", SimpleImputer(strategy="median"))
        ]
    )
    return prep

def build_prep_3(n_comp=None) -> Pipeline:
    log_cols = [
        'var3',
        'saldo_var30',
        'saldo_var42',
        'saldo_medio_var5_hace2',
        'saldo_medio_var5_hace3',
        'saldo_medio_var5_ult1',
        'saldo_medio_var5_ult3',
        'num_var42_0',
        'sum_of_saldo',
        'var38',
        'sum_of_num',
        'non_zero_count_num',
        'non_zero_count_ind'
    ]
    
    cat_cols = ["var36"]

    cat_tf = Pipeline(
        steps=[
            ("ohe", OneHotEncoder(min_frequency=100, sparse_output=False)),
        ]
    )

    prep = Pipeline(
        steps=[
            ("prep", build_prep()[:-2]),
            ("drop_almost", DropConstantColumns(thresh=.4, search=0)),
            ("log", CustomLog(columns = log_cols)),
            ("cat",ColumnTransformer([("ohe", cat_tf, cat_cols)], remainder='passthrough')),
            ("ss", StandardScaler()),
            ("knn", KNNImputer(n_neighbors=5)),
            ("pca", PCA(n_components=n_comp))
        ]
    )

    return prep