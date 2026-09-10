#!/usr/bin/env python3
"""8-node NCCL all-reduce benchmark over RoCE. One GPU per node."""
import os, time, torch, torch.distributed as dist

def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    if rank == 0:
        print(f"  world_size={world}  torch={torch.__version__}  "
              f"nccl={torch.cuda.nccl.version()}", flush=True)
        print(f"\n  {'size':>10}  {'time':>9}  {'algbw':>10}  {'busbw':>10}", flush=True)
        print("  " + "-" * 45, flush=True)

    # busbw factor for ring all-reduce: each byte crosses the wire 2(n-1)/n times
    factor = 2.0 * (world - 1) / world

    for mb in (8, 32, 128, 512, 1024):
        n = mb * 1024 * 1024 // 4                      # fp32 elements
        buf = torch.ones(n, dtype=torch.float32, device="cuda")

        for _ in range(3):                             # warmup
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()

        iters = 20 if mb <= 128 else 8
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        nbytes = n * 4
        algbw = nbytes / dt / 1e9
        busbw = algbw * factor
        if rank == 0:
            print(f"  {mb:>7} MB  {dt*1e3:>7.2f}ms  {algbw:>7.2f} GB/s  "
                  f"{busbw:>7.2f} GB/s", flush=True)
        del buf
        torch.cuda.empty_cache()

    # correctness: every rank contributed ones, so sum must equal world_size
    check = torch.ones(1024, device="cuda")
    dist.all_reduce(check)
    ok = bool((check == world).all().item())
    if rank == 0:
        print(f"\n  correctness: {'PASS' if ok else 'FAIL'} "
              f"(expected all elements == {world})", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
