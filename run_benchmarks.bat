@echo off
REM ============================================================
REM  Dataproc Benchmark Runner
REM  Runs all workloads with 1 (single-node), 2, 4, and 8 workers
REM  Each workload result saved to reports\benchmarks\
REM ============================================================

set REGION=europe-west1
set CLUSTER=egd-cluster
set MACHINE=n2-standard-2
set IMAGE=2.2-debian12
set DISK=50
set RUNS=3
set OUTDIR=reports\benchmarks

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo ============================================================
echo  DATAPROC BENCHMARK SUITE
echo  Machine: %MACHINE%   Runs: %RUNS%
echo  Configs: single-node, 2 workers, 4 workers, 8 workers
echo ============================================================

REM ============================================================
REM  PHASE 1: Single-node (1 node = master only)
REM ============================================================
echo.
echo [PHASE 1] Creating single-node cluster...
call gcloud dataproc clusters create %CLUSTER% --region=%REGION% --single-node --master-machine-type=%MACHINE% --image-version=%IMAGE% --master-boot-disk-size=%DISK%

echo.
echo [1 node] Running query_seasonality...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload query_seasonality --num-workers 1 --runs %RUNS% > "%OUTDIR%\query_seasonality_1w.txt" 2>&1
echo   Done.

echo.
echo [1 node] Running train_rf...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload train_rf --num-workers 1 --runs %RUNS% > "%OUTDIR%\train_rf_1w.txt" 2>&1
echo   Done.

echo.
echo [1 node] Running review_demand...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload review_demand --num-workers 1 --runs %RUNS% > "%OUTDIR%\review_demand_1w.txt" 2>&1
echo   Done.

echo.
echo [1 node] Running calendar_revenue...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload calendar_revenue --num-workers 1 --runs %RUNS% > "%OUTDIR%\calendar_revenue_1w.txt" 2>&1
echo   Done.

echo.
echo [1 node] Running full_pipeline...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload full_pipeline --num-workers 1 --runs %RUNS% > "%OUTDIR%\full_pipeline_1w.txt" 2>&1
echo   Done.

echo.
echo Deleting single-node cluster...
call gcloud dataproc clusters delete %CLUSTER% --region=%REGION% --quiet

REM ============================================================
REM  PHASE 2: 2 workers
REM ============================================================
echo.
echo [PHASE 2] Creating cluster with 2 workers...
call gcloud dataproc clusters create %CLUSTER% --region=%REGION% --num-workers=2 --master-machine-type=%MACHINE% --worker-machine-type=%MACHINE% --image-version=%IMAGE% --master-boot-disk-size=%DISK% --worker-boot-disk-size=%DISK%

echo.
echo [2 workers] Running query_seasonality...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload query_seasonality --num-workers 2 --runs %RUNS% > "%OUTDIR%\query_seasonality_2w.txt" 2>&1
echo   Done.

echo.
echo [2 workers] Running train_rf...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload train_rf --num-workers 2 --runs %RUNS% > "%OUTDIR%\train_rf_2w.txt" 2>&1
echo   Done.

echo.
echo [2 workers] Running review_demand...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload review_demand --num-workers 2 --runs %RUNS% > "%OUTDIR%\review_demand_2w.txt" 2>&1
echo   Done.

echo.
echo [2 workers] Running calendar_revenue...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload calendar_revenue --num-workers 2 --runs %RUNS% > "%OUTDIR%\calendar_revenue_2w.txt" 2>&1
echo   Done.

echo.
echo [2 workers] Running full_pipeline...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload full_pipeline --num-workers 2 --runs %RUNS% > "%OUTDIR%\full_pipeline_2w.txt" 2>&1
echo   Done.

echo.
echo Deleting 2-worker cluster...
call gcloud dataproc clusters delete %CLUSTER% --region=%REGION% --quiet

REM ============================================================
REM  PHASE 3: 4 workers
REM ============================================================
echo.
echo [PHASE 3] Creating cluster with 4 workers...
call gcloud dataproc clusters create %CLUSTER% --region=%REGION% --num-workers=4 --master-machine-type=%MACHINE% --worker-machine-type=%MACHINE% --image-version=%IMAGE% --master-boot-disk-size=%DISK% --worker-boot-disk-size=%DISK%

echo.
echo [4 workers] Running query_seasonality...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload query_seasonality --num-workers 4 --runs %RUNS% > "%OUTDIR%\query_seasonality_4w.txt" 2>&1
echo   Done.

echo.
echo [4 workers] Running train_rf...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload train_rf --num-workers 4 --runs %RUNS% > "%OUTDIR%\train_rf_4w.txt" 2>&1
echo   Done.

echo.
echo [4 workers] Running review_demand...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload review_demand --num-workers 4 --runs %RUNS% > "%OUTDIR%\review_demand_4w.txt" 2>&1
echo   Done.

echo.
echo [4 workers] Running calendar_revenue...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload calendar_revenue --num-workers 4 --runs %RUNS% > "%OUTDIR%\calendar_revenue_4w.txt" 2>&1
echo   Done.

echo.
echo [4 workers] Running full_pipeline...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload full_pipeline --num-workers 4 --runs %RUNS% > "%OUTDIR%\full_pipeline_4w.txt" 2>&1
echo   Done.

echo.
echo Deleting 4-worker cluster...
call gcloud dataproc clusters delete %CLUSTER% --region=%REGION% --quiet

REM ============================================================
REM  PHASE 4: 8 workers (may fail due to quota)
REM ============================================================
echo.
echo [PHASE 4] Creating cluster with 8 workers (may fail due to quota)...
call gcloud dataproc clusters create %CLUSTER% --region=%REGION% --num-workers=8 --master-machine-type=%MACHINE% --worker-machine-type=%MACHINE% --image-version=%IMAGE% --master-boot-disk-size=%DISK% --worker-boot-disk-size=%DISK%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: 8-worker cluster failed due to quota. Skipping.
    goto :done
)

echo.
echo [8 workers] Running query_seasonality...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload query_seasonality --num-workers 8 --runs %RUNS% > "%OUTDIR%\query_seasonality_8w.txt" 2>&1
echo   Done.

echo.
echo [8 workers] Running train_rf...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload train_rf --num-workers 8 --runs %RUNS% > "%OUTDIR%\train_rf_8w.txt" 2>&1
echo   Done.

echo.
echo [8 workers] Running review_demand...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload review_demand --num-workers 8 --runs %RUNS% > "%OUTDIR%\review_demand_8w.txt" 2>&1
echo   Done.

echo.
echo [8 workers] Running calendar_revenue...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload calendar_revenue --num-workers 8 --runs %RUNS% > "%OUTDIR%\calendar_revenue_8w.txt" 2>&1
echo   Done.

echo.
echo [8 workers] Running full_pipeline...
call gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=%CLUSTER% --region=%REGION% -- --workload full_pipeline --num-workers 8 --runs %RUNS% > "%OUTDIR%\full_pipeline_8w.txt" 2>&1
echo   Done.

echo.
echo Deleting 8-worker cluster...
call gcloud dataproc clusters delete %CLUSTER% --region=%REGION% --quiet

:done
echo.
echo ============================================================
echo  ALL BENCHMARKS COMPLETE!
echo  Results in: %OUTDIR%\
echo ============================================================
