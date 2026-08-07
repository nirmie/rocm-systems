////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in the
//    documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////
//
// OnDispatch HotSwap: dispatch-time, per-kernel cross-generation transpilation.
//
// Rather than transpiling an entire gfx1250 code object at load time (thousands
// of kernels, of which a workload runs only a handful), OnDispatch defers the
// work to the moment a kernel is actually dispatched:
//
//   Load:     hand the device a cheap loadable "tag" -- the original ELF
//             relabeled to the device ISA with every kernel body replaced by an
//             s_trap. CLR resolves valid kernel_object handles from it. Record
//             kernel_object -> {original source object, kernel name}.
//   Dispatch: an intercept queue reads each AQL packet's kernel_object; on first
//             sight it transpiles that one kernel, loads the real target-ISA
//             code, and swaps the packet's kernel_object to the real handle.
//             Subsequent dispatches are a table lookup.
//
// This header exposes the pieces as small, GPU-free units so the core logic can
// be unit tested offline (see hotswap_dispatch_test.cc). Only the wiring into
// the runtime's queue-create and load paths touches live HSA state.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_HOTSWAP_DISPATCH_HPP_
#define HSA_RUNTIME_CORE_INC_HOTSWAP_DISPATCH_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "inc/hsa.h"

// Runtime-integration headers are pulled in at file scope (never inside a
// namespace) so their own includes -- <optional>, etc. -- resolve correctly.
// The pure/offline unit-test build leaves this macro undefined and needs none
// of them.
#ifdef HSA_RUNTIME_CORE_HOTSWAP_DISPATCH_RUNTIME
#include "inc/hsa_api_trace.h"   // hsa_amd_queue_intercept_packet_writer
#include "core/inc/hotswap.hpp"  // CodeObjectView, LoadCodeObjectWithSizeFn
#endif

namespace rocr {
namespace hotswap {
namespace dispatch {

using OwnedElfBuffer = std::unique_ptr<void, decltype(&std::free)>;

// ---- Target ISA mach codes (llvm/BinaryFormat/ELF.h, EF_AMDGPU_MACH) --------
// Low byte of the ELF64 e_flags field. Only the cross-gen device targets are
// listed; a source object is always gfx1250 here.
constexpr uint32_t kEfAmdgpuMachMask = 0xff;
constexpr uint32_t kMachGfx942 = 0x04c;
constexpr uint32_t kMachGfx950 = 0x04f;
// ELF64 e_flags file offset: e_ident(16)+e_type(2)+e_machine(2)+e_version(4)
// +e_entry(8)+e_phoff(8)+e_shoff(8) = 48.
constexpr size_t kElf64EFlagsOffset = 48;

// Return the EF_AMDGPU_MACH code for a supported cross-gen device target
// ("gfx942"->0x4c, "gfx950"->0x4f), or 0 if unsupported. Pure.
uint32_t MachForTarget(const std::string& gfx_target);

// Produce a loadable "tag" for a gfx1250 code object on a cross-gen device:
// a COPY of `src` with (a) e_flags mach relabeled to `target_mach` and (b) the
// .text section overwritten with the target's `s_trap; s_endpgm` pattern. KDs
// (.rodata), symbol names, and metadata are preserved so the loader produces
// valid kernel_object handles. Returns an owned malloc'd buffer of `*out_size`
// bytes, or nullptr on malformed input. Pure byte manipulation; no HSA calls.
void* MakeRelabeledTrapCopy(const void* src, size_t size, uint32_t target_mach,
                            size_t* out_size);

// ---- Kernel swap table ------------------------------------------------------
// Maps the tag's kernel_object handle (what CLR resolved and will dispatch) to
// the information needed to transpile-on-first-dispatch, then caches the real
// target-ISA kernel_object once available. Thread-safe.
struct KernelEntry {
  // Original (gfx1250) whole code-object bytes, shared across all kernels from
  // the same object. Used as transpile input on first dispatch.
  std::shared_ptr<std::vector<uint8_t>> source_object;
  std::string name;          // mangled kernel symbol name
  uint64_t real_kernel_object = 0;  // 0 until transpiled+loaded
  // Scratch (private) and LDS (group) segment sizes of the real transpiled
  // kernel, read from its executable symbol when it is loaded. The dispatch
  // packet HIP built refers to the tag stub (which declares 0 for both), so the
  // interceptor must overwrite these fields alongside kernel_object; otherwise a
  // transpiled kernel that needs scratch runs under a packet declaring none and
  // the command processor rejects the dispatch (REGISTER_SIZE_INVALID).
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
};

class KernelSwapTable {
 public:
  void Insert(uint64_t tag_kernel_object,
              std::shared_ptr<std::vector<uint8_t>> source_object,
              std::string name);

  // Look up the entry for a tag handle. Returns nullptr if unknown (not one of
  // ours -> pass through untouched).
  bool Lookup(uint64_t tag_kernel_object, KernelEntry* out) const;

  // Record the real target-ISA handle (and its scratch/LDS segment sizes) after
  // a successful transpile+load. Sizes default to 0 for callers (e.g. pure-logic
  // unit tests) that only exercise handle swapping.
  void SetReal(uint64_t tag_kernel_object, uint64_t real_kernel_object,
               uint32_t private_segment_size = 0,
               uint32_t group_segment_size = 0);

  // Resolve a dispatched handle to the handle that should actually run:
  // - unknown handle            -> returns `tag_kernel_object` (pass through)
  // - known and already real    -> returns the real handle
  // - known but not yet real    -> returns 0 (caller must transpile first)
  uint64_t Resolve(uint64_t tag_kernel_object) const;

  // All (tag_kernel_object, name) entries whose source object is `source`.
  // Used after a side executable is loaded to wire every kernel from that
  // object in one pass.
  std::vector<std::pair<uint64_t, std::string>> EntriesForSource(
      const void* source) const;

  size_t Size() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, KernelEntry> map_;
};

// ---- AQL packet rewrite (pure) ----------------------------------------------
// Walk `packet_count` AQL packets at `packets` (each sizeof(hsa_kernel_dispatch
// _packet_t) = 64 bytes). For every KERNEL_DISPATCH packet whose kernel_object
// resolves to a real handle in `table`, rewrite the field in place. Packets
// that are not kernel dispatches, or whose handle is unknown/not-yet-real, are
// left untouched. Returns the number of packets whose kernel_object was
// swapped. Pure: operates on a plain buffer, no queue/GPU. This is the core of
// the dispatch interceptor and is what the unit tests exercise.
size_t RewritePackets(void* packets, uint64_t packet_count,
                      const KernelSwapTable& table);

// ---- Runtime integration (on-device) ----------------------------------------
// These tie the pure pieces above into the live runtime. They are only active
// when OnDispatch is enabled; see IsOnDispatchEnabled. The declarations that
// reference the tools-only intercept ABI are guarded so the pure logic above
// can be compiled and unit-tested without hsa_api_trace.h.

// OnDispatch is opt-in via env HSA_HOTSWAP_ONDISPATCH (default off), so the
// normal load-time HotSwap path is unchanged unless explicitly requested.
bool IsOnDispatchEnabled();

// Process-wide swap table shared by the load path (populates it) and the queue
// interceptor (consults it).
KernelSwapTable& GlobalSwapTable();

// Transpile+load the real target-ISA kernel for a tag handle if not already
// done, recording orig->real in the table. Returns the real kernel_object, or
// 0 on failure (caller leaves the trap in place -> loud fault, never silent
// wrong-ISA execution). Requires COMGR; only called on-device.
uint64_t EnsureRealKernel(hsa_agent_t agent, uint64_t tag_kernel_object);

#ifdef HSA_RUNTIME_CORE_HOTSWAP_DISPATCH_RUNTIME
// The AQL dispatch interceptor registered on every intercept queue. Matches the
// hsa_amd_queue_intercept_handler ABI (hsa_api_trace.h). On first sight of a
// not-yet-transpiled kernel it synchronously transpiles+loads that kernel via
// EnsureRealKernel, then rewrites the packet's kernel_object and forwards via
// `writer`.
void DispatchInterceptor(const void* packets, uint64_t packet_count,
                         uint64_t user_pkt_index, void* data,
                         hsa_amd_queue_intercept_packet_writer writer);

// Load-time entry for the OnDispatch path. If `code_object` is a supported
// cross-gen source (gfx1250 -> device gfx942/gfx950), load a relabel+trap tag
// via `load_sized` (signature of LoadSizedCodeObject), record each loaded
// kernel_object -> {source bytes, name} in the global swap table, and set
// *out_status to the load result; returns true (handled). Returns false when
// this object is not an OnDispatch cross-gen case, so the caller runs the
// normal load path. `load_sized` and `context` are the runtime's
// LoadSizedCodeObject callback and Executable*.
bool TryLoadTagForOnDispatch(
    hsa_executable_t executable, hsa_agent_t agent,
    const CodeObjectView& code_object, const char* options,
    hsa_loaded_code_object_t* loaded_code_object,
    LoadCodeObjectWithSizeFn load_sized, void* context,
    hsa_status_t* out_status);
#endif  // HSA_RUNTIME_CORE_HOTSWAP_DISPATCH_RUNTIME

}  // namespace dispatch
}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_DISPATCH_HPP_
