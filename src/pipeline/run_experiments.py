"""
Master script to run all forecasting experiments across scenarios.

Usage:
    # Run all experiments for all scenarios
    python run_all_experiments.py

    # Run specific scenario
    python run_all_experiments.py --scenario A

    # Run specific models
    python run_all_experiments.py --models prophet arima var

    # Use Auto-ARIMA instead of manual ARIMA
    python run_all_experiments.py --auto-arima

    # Run only ML models
    python run_all_experiments.py --models random_forest xgboost lightgbm
"""

import subprocess
import argparse
import time
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(model_script: str, scenario: str, extra_args: list = None) -> dict:
    """
    Run a single experiment and return results.

    Args:
        model_script: Path to model script (e.g., 'models/prophet_model.py')
        scenario: 'A', 'B', or 'C'
        extra_args: Additional command line arguments (e.g., ['--auto'] for ARIMA)

    Returns:
        dict with status, runtime, and results path
    """
    cmd = ['python', '-m', model_script.replace('/', '.').replace('.py', ''), scenario]
    if extra_args:
        cmd.extend(extra_args)

    model_name = os.path.basename(model_script).replace('_model.py', '')
    exp_name = model_name
    if extra_args and '--auto' in extra_args:
        exp_name = f"{model_name}_auto"

    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {exp_name} - Scenario {scenario}")
    logger.info(f"{'='*80}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=43200  # 12 hour timeout
        )

        runtime = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"✓ Completed in {runtime/60:.1f} minutes")

            # Determine results path
            results_file = f"results/{exp_name}_scenario_{scenario}_results.json"

            return {
                'status': 'success',
                'runtime': runtime,
                'results_path': results_file,
                'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout  # Last 1000 chars
            }
        else:
            logger.error(f"✗ Failed with return code {result.returncode}")
            logger.error(f"Error output:\n{result.stderr[-1000:]}")

            return {
                'status': 'failed',
                'runtime': runtime,
                'error': result.stderr[-1000:]
            }

    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        logger.error(f"✗ Timeout after {runtime/3600:.1f} hours")

        return {
            'status': 'timeout',
            'runtime': runtime
        }

    except Exception as e:
        runtime = time.time() - start_time
        logger.error(f"✗ Exception: {e}")

        return {
            'status': 'error',
            'runtime': runtime,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run all forecasting experiments')
    parser.add_argument('--scenario', type=str, choices=['A', 'B', 'C', 'all'], default='all',
                       help='Scenario to run (default: all)')
    parser.add_argument('--models', nargs='+',
                       choices=['prophet', 'arima', 'var', 'random_forest', 'xgboost', 'lightgbm', 'all'],
                       default=['all'],
                       help='Models to run (default: all)')
    parser.add_argument('--auto-arima', action='store_true',
                       help='Use Auto-ARIMA for automatic order selection (requires pmdarima)')
    parser.add_argument('--output', type=str, default='experiment_log.json',
                       help='Output file for experiment log')

    args = parser.parse_args()

    # Determine scenarios to run
    scenarios = ['A', 'B', 'C'] if args.scenario == 'all' else [args.scenario]

    # Determine models to run
    if 'all' in args.models:
        models = ['prophet', 'arima', 'var', 'random_forest', 'xgboost', 'lightgbm']
    else:
        models = args.models

    # Build experiment list
    experiments = []

    for scenario in scenarios:
        for model in models:
            if model == 'arima':
                # ARIMA model with optional auto-arima
                experiments.append({
                    'model': 'arima_auto' if args.auto_arima else 'arima',
                    'script': f'models/{model}_model.py',
                    'scenario': scenario,
                    'extra_args': ['--auto'] if args.auto_arima else None
                })
            elif model in ['prophet', 'var']:
                # Other statistical models (direct forecasting)
                experiments.append({
                    'model': model,
                    'script': f'models/{model}_model.py',
                    'scenario': scenario,
                    'extra_args': None
                })
            else:
                # ML models (direct multi-horizon approach)
                experiments.append({
                    'model': model,
                    'script': f'models/{model}_model.py',
                    'scenario': scenario,
                    'extra_args': None
                })

    # Run experiments
    logger.info(f"\n{'='*80}")
    logger.info(f"Running {len(experiments)} experiments")
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Models: {models}")
    logger.info(f"Auto-ARIMA: {args.auto_arima}")
    logger.info(f"{'='*80}\n")

    results = []
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        logger.info(f"\n[{i}/{len(experiments)}] Starting experiment...")

        result = run_experiment(exp['script'], exp['scenario'], exp.get('extra_args'))

        results.append({
            'experiment': exp,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })

        # Save progress after each experiment
        with open(args.output, 'w') as f:
            json.dump({
                'experiments': results,
                'total_runtime': time.time() - total_start,
                'completed': i,
                'total': len(experiments)
            }, f, indent=2)

    total_runtime = time.time() - total_start

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")

    successful = sum(1 for r in results if r['result']['status'] == 'success')
    failed = sum(1 for r in results if r['result']['status'] == 'failed')
    timeout = sum(1 for r in results if r['result']['status'] == 'timeout')
    errors = sum(1 for r in results if r['result']['status'] == 'error')

    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Timeout: {timeout}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total runtime: {total_runtime/3600:.2f} hours")

    logger.info(f"\nDetailed results saved to: {args.output}")

    # Print per-experiment results
    logger.info(f"\nPer-experiment runtimes:")
    for r in results:
        exp = r['experiment']
        result = r['result']
        exp_name = exp['model']  # Already includes 'arima_auto' when auto-arima is used
        status_icon = '✓' if result['status'] == 'success' else '✗'
        runtime_str = f"{result['runtime']/60:.1f}min"

        logger.info(f"  {status_icon} {exp_name:30s} Scenario {exp['scenario']:1s}: {runtime_str:10s} [{result['status']}]")

    # Create results summary table
    if successful > 0:
        logger.info(f"\n{'='*80}")
        logger.info("MODEL PERFORMANCE COMPARISON")
        logger.info(f"{'='*80}\n")

        for scenario in scenarios:
            logger.info(f"Scenario {scenario}:")
            logger.info(f"{'Model':<30s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'MAPE':>8s} {'MASE':>8s}")
            logger.info("-" * 80)

            for r in results:
                if r['experiment']['scenario'] == scenario and r['result']['status'] == 'success':
                    exp = r['experiment']
                    exp_name = exp['model']  # Already includes 'arima_auto' when auto-arima is used

                    try:
                        with open(r['result']['results_path']) as f:
                            data = json.load(f)
                            test_metrics = data['test']['aggregate']

                            logger.info(
                                f"{exp_name:<30s} "
                                f"{test_metrics['rmse']:>8.4f} "
                                f"{test_metrics['mae']:>8.4f} "
                                f"{test_metrics['r2']:>8.4f} "
                                f"{test_metrics['mape']:>8.2f} "
                                f"{test_metrics['mase']:>8.4f}"
                            )
                    except Exception as e:
                        logger.warning(f"Could not load results for {exp_name}: {e}")

            logger.info("")

    logger.info(f"\n{'='*80}")
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
