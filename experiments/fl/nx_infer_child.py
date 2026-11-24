"""NX's inference child (runs/20261105-fx/plan.md, FX review minor 4): the 9 GiB device cap of every other unit of this line, then experiments/nn/infer_nn.py itself, unchanged.  Runs in the conda environment nnunet as `python -I -B nx_infer_child.py <infer_nn.py> <its arguments>`; imports nothing of this repository.

infer_nn.py belongs to work package NN and is not edited here: the cap is torch.cuda.set_per_process_memory_fraction before the script runs, in the same process.  Over the cap the caching allocator raises an out-of-memory error.  nnunetv2 2.8.1 (predict_from_raw_data.py, predict_sliding_window_return_logits) catches the first one when it predicts on the device: it prints "Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU" and computes that subject again with the results array on the CPU, the peak still under the cap.  Only a second error on the same subject escapes: exit 1, the queue stops (FX review follow-up, minor A).  run_fx.nx_infer_one finds that line in infer.log and writes which subjects fell back into infer_check.json; infer_record.json keeps the peaks (gpu_peak_allocated_gib, gpu_peak_reserved_gib).
"""
import runpy
import sys

LIMIT_BYTES = 9*2**30


def main():
    script, argv = sys.argv[1], sys.argv[2:]
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., LIMIT_BYTES/total), 0)
        print(f'device cap {LIMIT_BYTES/2**30:g} GiB of {total/2**30:.1f} GiB', flush=True)
    sys.argv = [script, *argv]
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
