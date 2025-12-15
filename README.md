# python-practice
Contains my python practice files

# --- 0) Spark + imports -------------------------------------------------------
# NOTE: This extends the earlier working example.
# ADDED: CrossValidator, ParamGridBuilder, error handling, and parallelization knobs.

from pyspark.sql import SparkSession
spark = (SparkSession.builder
         .appName("optuna-synapseml-lightgbm-with-cv")
         # If needed, add SynapseML package:
         # .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.11.2")
         .getOrCreate())

from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from synapse.ml.lightgbm import LightGBMClassifier  # <- SynapseML estimator

import optuna
from optuna.exceptions import TrialPruned

import pandas as pd
from sklearn.datasets import make_classification
import math
import traceback

# --- 0.1) Global knobs (parallelism, CV, safety) -----------------------------
SEED = 7

# (A) Control LightGBM's per-executor thread parallelism (intra-executor)
#     This sets the OpenMP threads per LightGBM worker.
LGBM_NUM_THREADS = 4  # tune per your executor cores; 1..N

# (B) Control Spark CrossValidator parallelism (evaluates grid points concurrently)
CV_NUM_FOLDS = 3
CV_PARALLELISM = 2  # safe value; increase if cluster has spare cores

# (C) Optuna HPO controls
HPO_N_TRIALS = 30
HPO_N_JOBS = 1  # KEEP 1 by default. >1 means multiple fits concurrently from same SparkSession (fragile).
USE_OPTUNA_PRUNER = True  # enables pruning on bad intermediate results

# (D) Safety: small, local grid per trial for the CrossValidator (keeps runtime in check)
GRID_SPAN_FRAC = 0.15  # ±15% around trial suggestion for fractional params
GRID_NEIGHBORS_INT = 1  # integer neighborhood size around trial suggestion (e.g., +/-1)

# --- 1) Build or load a Spark DataFrame with features + label -----------------
# Example dataset (replace with your own Spark DataFrame "df" with 'label' + feature columns)
X, y = make_classification(
    n_samples=20000, n_features=40, n_informative=10, n_redundant=10,
    n_repeated=0, n_classes=2, weights=[0.6, 0.4], random_state=SEED
)
pdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
pdf["label"] = y.astype("int32")
df = spark.createDataFrame(pdf)

feature_cols = [c for c in df.columns if c != "label"]

# --- 2) Train/valid split for OUT-OF-FOLD evaluation -------------------------
# NOTE: CrossValidator will do its own k-fold on the *training* partition.
# We preserve a small VALIDATION HOLDOUT to compute the final metric per trial.
train_df, valid_df = df.randomSplit([0.8, 0.2], seed=SEED)

# Performance: persist once and reuse
train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
valid_df = valid_df.persist(StorageLevel.MEMORY_AND_DISK)
_ = train_df.count(); _ = valid_df.count()  # materialize

# (Optional but recommended) checkpointing for complex pipelines
# spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

# --- 3) Shared pipeline pieces ------------------------------------------------
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

# --- 3.1) Edge-case guarding: safe folds for CV ------------------------------
def safe_cv_folds(df, requested_folds: int) -> int:
    """Reduce folds if dataset is too small or highly imbalanced to support requested_folds."""
    total = df.count()
    if total < requested_folds:
        return max(2, min(3, total))  # fall back to 2 or 3
    # crude class counts (avoid expensive groupBy on every trial by computing once here)
    pos = df.filter(F.col("label") == 1).count()
    neg = total - pos
    min_class = min(pos, neg)
    # need at least 1 example of each class per fold
    if min_class < requested_folds:
        return max(2, min(3, min_class))  # reduce folds to fit minority class
    return requested_folds

SAFE_CV_NUM_FOLDS = safe_cv_folds(train_df, CV_NUM_FOLDS)

# --- 4) Optuna objective with nested Spark CrossValidator --------------------
def _bounded_frac(center: float, span: float):
    lo = max(0.05, center * (1.0 - span))
    hi = min(0.999, center * (1.0 + span))
    # ensure uniqueness & sorted
    return sorted({round(lo, 6), round(center, 6), round(hi, 6)})

def _neighbors_int(center: int, k: int, lo: int, hi: int):
    vals = set([center])
    for d in range(1, k + 1):
        if center - d >= lo: vals.add(center - d)
        if center + d <= hi: vals.add(center + d)
    return sorted(vals)

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective:
    - Suggest a set of LightGBM hyperparams.
    - Build a *small* CV grid around a subset (featureFraction, baggingFraction, numLeaves).
    - Run Spark CrossValidator on the training partition.
    - Evaluate best CV model on the hold-out valid_df (AUC).
    - Return AUC (we set study to maximize), with robust error handling.
    """
    # ADDED: pruner (optional)
    if USE_OPTUNA_PRUNER:
        trial.report(0.0, step=0)  # seed an initial report so pruner can kick in

    try:
        # --- 4.1) Sample base hyperparameters (camelCase for SynapseML) ------
        numLeaves = trial.suggest_int("numLeaves", 32, 512, log=True)
        maxDepth = trial.suggest_int("maxDepth", -1, 16)
        learningRate = trial.suggest_float("learningRate", 1e-3, 0.3, log=True)
        minSumHessianInLeaf = trial.suggest_float("minSumHessianInLeaf", 1e-3, 10.0, log=True)
        lambdaL1 = trial.suggest_float("lambdaL1", 0.0, 10.0)
        lambdaL2 = trial.suggest_float("lambdaL2", 0.0, 10.0)
        featureFraction = trial.suggest_float("featureFraction", 0.5, 1.0)
        baggingFraction = trial.suggest_float("baggingFraction", 0.5, 1.0)
        baggingFreq = trial.suggest_int("baggingFreq", 0, 10)
        numIterations = trial.suggest_int("numIterations", 300, 1500)  # CV w/o early stopping: keep bounded
        # ADDED: per-trial training parallelization control
        numThreads = trial.suggest_int("numThreads", 1, max(1, LGBM_NUM_THREADS))  # per-exec threads (<= cap)

        # --- 4.2) Define the base estimator ---------------------------------
        lgbm = (LightGBMClassifier(
                    objective="binary",
                    featuresCol="features",
                    labelCol="label",
                    # NOTE: Do NOT set validationIndicatorCol inside CV (CV manages its own folds).
                    numLeaves=numLeaves,
                    maxDepth=maxDepth,
                    minSumHessianInLeaf=minSumHessianInLeaf,
                    lambdaL1=lambdaL1,
                    lambdaL2=lambdaL2,
                    learningRate=learningRate,
                    featureFraction=featureFraction,
                    baggingFraction=baggingFraction,
                    baggingFreq=baggingFreq,
                    numIterations=numIterations,
                    numThreads=numThreads,
                    seed=SEED,
                    # Stability knobs (optional):
                    # useBarrierExecution=True,  # enable on K8s/YARN if you see hanging tasks
                    # chunkSize=10,
                    # numBatches=0
               ))

        pipeline = Pipeline(stages=[assembler, lgbm])

        # --- 4.3) Build a *local* grid around trial suggestions --------------
        # Keep grid small to avoid exploding runtime. Each grid point is a full distributed train.
        ff_grid = _bounded_frac(featureFraction, GRID_SPAN_FRAC)  # around featureFraction
        bf_grid = _bounded_frac(baggingFraction, GRID_SPAN_FRAC)  # around baggingFraction
        nl_grid = _neighbors_int(numLeaves, GRID_NEIGHBORS_INT, lo=16, hi=1024)  # neighborhood around numLeaves

        paramGrid = (ParamGridBuilder()
                     .addGrid(lgbm.featureFraction, ff_grid)
                     .addGrid(lgbm.baggingFraction, bf_grid)
                     .addGrid(lgbm.numLeaves, nl_grid)
                     .build())

        # --- 4.4) CrossValidator on training split ---------------------------
        # Parallelism here controls how many grid points run simultaneously (each a Spark job).
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=SAFE_CV_NUM_FOLDS,
            parallelism=CV_PARALLELISM,
            seed=SEED
        )

        cv_model = cv.fit(train_df)  # k-fold CV on train partition only

        # --- 4.5) Evaluate best model on holdout validation ------------------
        best_model = cv_model.bestModel
        preds_valid = best_model.transform(valid_df)
        auc = evaluator.evaluate(preds_valid)

        # ADDED: report to Optuna (enables pruning if configured)
        trial.report(auc, step=1)
        if USE_OPTUNA_PRUNER and trial.should_prune():
            raise TrialPruned(f"Pruned (AUC={auc:.4f})")

        return auc  # Study set to 'maximize' below

    except TrialPruned as tp:
        # Proper pruning: allows Optuna to drop unpromising trials early.
        raise tp
    except Exception as e:
        # Robust error handling: capture trace for later inspection and prune.
        trial.set_user_attr("exception", repr(e))
        trial.set_user_attr("traceback", traceback.format_exc())
        # Prune rather than return a poisonous metric; keeps study healthy.
        raise TrialPruned(f"Pruned due to exception: {repr(e)}")

# --- 5) Run the study (sequential by default for Spark safety) ----------------
sampler = optuna.samplers.TPESampler(seed=SEED)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=5) if USE_OPTUNA_PRUNER else optuna.pruners.NopPruner()

study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="lgbm_optuna_cv_nested")

# IMPORTANT:
# - Keep n_jobs=1 unless you have a robust multi-process orchestration that isolates Spark fits.
# - If you set n_jobs>1, prefer *process-based* workers that each own their own SparkSession.
study.optimize(objective, n_trials=HPO_N_TRIALS, n_jobs=HPO_N_JOBS, show_progress_bar=True)

print("Best AUC (CV→valid):", study.best_value)
print("Best (trial-level) params:", study.best_params)

# --- 6) Final model training with EARLY STOPPING ------------------------------
# Now retrain on train+valid with validationIndicatorCol so LightGBM can early-stop.
# NOTE: We intentionally do *not* use CV here; we want early stopping on a fixed validation slice.

# Prepare union with validation indicator for early stopping
train_ev_df = (train_df.withColumn("val", F.lit(False))
               .unionByName(valid_df.withColumn("val", F.lit(True)))
               .persist(StorageLevel.MEMORY_AND_DISK))
_ = train_ev_df.count()

# Reuse best trial params as a baseline; you may also merge best grid values from cv_model if desired.
best = study.best_params.copy()
best.update({
    "objective": "binary",
    "featuresCol": "features",
    "labelCol": "label",
    "validationIndicatorCol": "val",
    "earlyStoppingRound": 50,        # typical default; adjust as needed
    "numIterations": max(500, best.get("numIterations", 500)),  # allow early stopping to kick in
    "numThreads": best.get("numThreads", LGBM_NUM_THREADS),
    "seed": SEED
})

final_lgbm = LightGBMClassifier(**best)
final_pipeline = Pipeline(stages=[assembler, final_lgbm])
final_model = final_pipeline.fit(train_ev_df)

# Optional: independent test sample to report a final metric
test_df = df.sample(withReplacement=False, fraction=0.2, seed=SEED)
preds_test = final_model.transform(test_df)
test_auc = evaluator.evaluate(preds_test)
print("Final Test AUC (early-stopped):", test_auc)
