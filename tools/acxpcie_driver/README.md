# Achronix `acxpcie` kernel driver -- local patches

The driver source of record on `fics` is `~/pi0_board_bundle/sdk/driver/src`
(a copy of `SDK/drivers/acxpcie` from the Achronix SDK v2.1.1, plus the
6.8-kernel build fixes made on 2026-09-07). The built module the board
scripts `insmod` is `~/pi0_board_bundle/sdk/driver/acxpcie.ko`. Neither is
tracked in this repo; the patches here are, so they survive a lost bundle.

## 0001 -- `acxdev_mmap()`: map DMA buffers with `dma_mmap_coherent()`

**Symptom.** Every DMA-path run of `llama3_runtime_bringup` on 2026-09-08
"hung silently" and printed nothing. It did not hang: the kernel killed it.
`journalctl -k -b -1` on fics has twelve of these, one per attempt, on both
the p190s3b and the p175bar bitstreams:

    llama3_runtime_: Corrupted page table at address 7a1aa8300000
    PGD 10929a067 P4D 10929a067 PUD 18c20a067 PMD 1079b5067 PTE 8000467c4a5c9237
    Bad pagetable: 000f [#1] PREEMPT SMP NOPTI
    RIP: 0033:0x7a1aa8fa08ac            <- user space, glibc memmove
    RDI: 00007a1aa8300000  RDX: 0000000000000010
    note: llama3_runtime_[6325] exited with irqs disabled
    BUG: scheduling while atomic: llama3_runtime_/6325/0x00000000

The faulting address is the freshly `mmap()`ed DMA buffer, the instruction is
the first 16-byte store into it, and the PTE's frame number (0x467c4a5c9)
lies above the CPU's physical address width, which is why the page-fault
handler reports a corrupted page table instead of a plain SIGSEGV. The task is
torn down inside the fault path, so no C++ exception, no `printf`, and the
parent shell keeps waiting on it.

**Cause.** `acxdev_mmap()` mapped DMA buffers with

    pfn = virt_to_phys(kva) >> PAGE_SHIFT;
    remap_pfn_range(vma, vma->vm_start, pfn, vma_size, vma->vm_page_prot);

That is only valid when `dma_alloc_coherent()` returns a linear-map address.
On fics the Intel IOMMU is on by default (`CONFIG_INTEL_IOMMU_DEFAULT_ON=y`,
`iommu: Default domain type: Translated`, the card in a DMA-FQ group), and in
a translated domain `dma_alloc_coherent()` returns a `vmap()` of pages that
need not be physically contiguous. `virt_to_phys()` of a vmalloc address is
garbage.

**Fix.** `dma_mmap_coherent(&pdev->dev, vma, kva, dma_addr, size)`, the API
that exists for exactly this, correct for both the remapped and the linear
case. Nothing else in the driver assumes physical contiguity: the `dma_addr`
handed to user space is the IOVA, which is what the DMA engine must use.

**Alternative, not taken.** Booting with `iommu=pt` (or `intel_iommu=off`)
would make `dma_alloc_coherent()` return linear-map memory and hide the bug.
Keep it in mind only if the IOMMU turns out to reject the DMA engine's own
transactions (that would show as `DMAR: ... fault addr ...` lines naming
`01:00.0` in `dmesg`, and none appeared on 2026-09-08).

### Apply, build, install

```bash
cd ~/pi0_board_bundle/sdk/driver/src
patch -p2 < ~/projects/pi0_achronix/tools/acxpcie_driver/0001-acxdev_mmap-use-dma_mmap_coherent.patch
make modules                      # needs /lib/modules/$(uname -r)/build
cp acxpcie.ko ../acxpcie.ko       # what host/run.sh users insmod
sudo rmmod acxpcie 2>/dev/null; sudo insmod ../acxpcie.ko
```

Done on fics on 2026-09-09: `~/pi0_board_bundle/sdk/driver/acxpcie.ko` is
the patched build (srcversion `1CF87CF71CDCD1A33A76FB9`, vermagic
6.8.0-138-generic); the previous module is kept beside it as
`acxpcie.ko.virt_to_phys_20260901`. The module loads and unloads cleanly with
no card present; the first load with the card is the real test -- after it,
`dmesg` must show no `Corrupted page table` and the runtime's first `h2d()`
must return.

### How to tell this apart from a real DMA hang

A process killed this way leaves `note: ... exited with irqs disabled` in
`dmesg`. A process that is genuinely waiting on the DMA engine does not, and
with `wait_or_halt` in the runtime it throws after the 10 s transfer timeout
with the controller's registers in the message. Also run the bring-up CLI
with `stdbuf -oL` when piping its output through `tee`: `printf` to a pipe is
fully buffered, and a killed process never flushes.
