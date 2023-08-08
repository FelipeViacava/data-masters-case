# --- Transformers --- #
from sklearn.impute import SimpleImputer
from resources.customtransformers import \
    DropConstantColumns, \
    DropDuplicateColumns, \
    AddNonZeroCount, \
    CustomSum, \
    CustomImputer, \
    AddNoneCount, \
    CustomEncoder

# --- Pipeline Building --- #
from sklearn.pipeline import Pipeline

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
            ("nan", SimpleImputer(strategy="median"))
        ]
    )
    return prep_nan