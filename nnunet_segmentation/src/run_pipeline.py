#!/usr/bin/env python3
import os
import sys


def main() -> None:
    # Ensure we run relative to this file so config.yaml is found
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Lazy imports so PYTHONPATH includes current dir
    from Step_1_data_preparation import main as step1_main
    from Step_2_data_preprocessing import main as step2_main
    from Step_3_run_segmentation import main as step3_main

    # Run the three steps
    step1_main()
    step2_main()
    step3_main()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        raise


