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

// This translation unit provides both the pure logic (also compiled standalone
// by the offline gtest, which defines HOTSWAP_DISPATCH_PURE_ONLY) and the
// runtime integration that links into libhsa-runtime. In the normal library
// build we enable the runtime section here so it is compiled exactly once.
#if !defined(HOTSWAP_DISPATCH_PURE_ONLY) && \
    !defined(HSA_RUNTIME_CORE_HOTSWAP_DISPATCH_RUNTIME)
#define HSA_RUNTIME_CORE_HOTSWAP_DISPATCH_RUNTIME
#endif

#include "core/inc/hotswap_dispatch.hpp"

#include <algorithm>
#include <cstring>

namespace rocr {
namespace hotswap {
namespace dispatch {

uint32_t MachForTarget(const std::string& gfx_target) {
  if (gfx_target == "gfx942") return kMachGfx942;
  if (gfx_target == "gfx950") return kMachGfx950;
  return 0;
}

namespace {

// Little-endian helpers over a byte buffer. All ELF64 fields we touch are
// naturally aligned within the file image, but we memcpy to stay strictly
// well-defined regardless of alignment.
template <typename T>
T ReadLE(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

template <typename T>
void WriteLE(uint8_t* p, T v) {
  std::memcpy(p, &v, sizeof(T));
}

// gfx9 (CDNA: gfx942/gfx950) trap pattern: `s_trap 2; s_endpgm`, 8 bytes LE.
// Any wavefront that reaches an un-swapped tag kernel halts loudly here instead
// of executing wrong-ISA bytes. Both instructions share the SOPP encoding form
// across gfx942 and gfx950.
constexpr uint8_t kTrapPattern[8] = {0x02, 0x00, 0x92, 0xbf,
                                     0x00, 0x00, 0x81, 0xbf};

// ELF64 section-header field offsets (within a 64-byte Elf64_Shdr).
constexpr size_t kShdrNameOff = 0;    // sh_name   (u32)
constexpr size_t kShdrTypeOff = 4;    // sh_type   (u32)
constexpr size_t kShdrOffsetOff = 24; // sh_offset (u64)
constexpr size_t kShdrSizeOff = 32;   // sh_size   (u64)
constexpr uint32_t kShtProgbits = 1;

// ELF64 header field offsets used to reach the section headers.
constexpr size_t kEhdrShoffOff = 40;     // e_shoff     (u64)
constexpr size_t kEhdrShentsizeOff = 58; // e_shentsize (u16)
constexpr size_t kEhdrShnumOff = 60;     // e_shnum     (u16)
constexpr size_t kEhdrShstrndxOff = 62;  // e_shstrndx  (u16)

}  // namespace

void* MakeRelabeledTrapCopy(const void* src_v, size_t size, uint32_t target_mach,
                            size_t* out_size) {
  if (out_size) *out_size = 0;
  const uint8_t* src = static_cast<const uint8_t*>(src_v);
  // Require an ELF64 magic and enough room for the ELF header.
  if (!src || size < 64 || target_mach == 0 || src[0] != 0x7f || src[1] != 'E' ||
      src[2] != 'L' || src[3] != 'F' || src[4] != 2 /*ELFCLASS64*/) {
    return nullptr;
  }

  auto* buf = static_cast<uint8_t*>(std::malloc(size));
  if (!buf) return nullptr;
  std::memcpy(buf, src, size);

  // (a) Relabel e_flags mach nibble to the device target.
  uint32_t eflags = ReadLE<uint32_t>(buf + kElf64EFlagsOffset);
  eflags = (eflags & ~kEfAmdgpuMachMask) | (target_mach & kEfAmdgpuMachMask);
  WriteLE<uint32_t>(buf + kElf64EFlagsOffset, eflags);

  // Locate the section headers.
  const uint64_t e_shoff = ReadLE<uint64_t>(buf + kEhdrShoffOff);
  const uint16_t e_shentsize = ReadLE<uint16_t>(buf + kEhdrShentsizeOff);
  const uint16_t e_shnum = ReadLE<uint16_t>(buf + kEhdrShnumOff);
  const uint16_t e_shstrndx = ReadLE<uint16_t>(buf + kEhdrShstrndxOff);
  if (e_shoff == 0 || e_shentsize < 64 || e_shnum == 0 ||
      e_shstrndx >= e_shnum ||
      e_shoff + static_cast<uint64_t>(e_shnum) * e_shentsize > size) {
    std::free(buf);
    return nullptr;
  }

  auto shdr_field64 = [&](uint16_t idx, size_t field_off) -> uint64_t {
    return ReadLE<uint64_t>(buf + e_shoff +
                            static_cast<uint64_t>(idx) * e_shentsize + field_off);
  };
  auto shdr_field32 = [&](uint16_t idx, size_t field_off) -> uint32_t {
    return ReadLE<uint32_t>(buf + e_shoff +
                            static_cast<uint64_t>(idx) * e_shentsize + field_off);
  };

  // Section-header string table, used to match sections by name.
  const uint64_t shstr_off = shdr_field64(e_shstrndx, kShdrOffsetOff);
  const uint64_t shstr_size = shdr_field64(e_shstrndx, kShdrSizeOff);
  if (shstr_off >= size || shstr_off + shstr_size > size) {
    std::free(buf);
    return nullptr;
  }

  // (b) Fill the single .text PROGBITS section with the trap pattern.
  bool filled = false;
  for (uint16_t i = 0; i < e_shnum; ++i) {
    if (shdr_field32(i, kShdrTypeOff) != kShtProgbits) continue;
    const uint32_t sh_name = shdr_field32(i, kShdrNameOff);
    if (shstr_off + sh_name + sizeof(".text") > size) continue;
    const char* nm = reinterpret_cast<const char*>(buf + shstr_off + sh_name);
    if (std::strncmp(nm, ".text", sizeof(".text")) != 0) continue;

    const uint64_t sh_off = shdr_field64(i, kShdrOffsetOff);
    const uint64_t sh_size = shdr_field64(i, kShdrSizeOff);
    if (sh_off > size || sh_off + sh_size > size) break;
    for (uint64_t off = 0; off < sh_size; off += sizeof(kTrapPattern)) {
      const size_t n = static_cast<size_t>(
          std::min<uint64_t>(sizeof(kTrapPattern), sh_size - off));
      std::memcpy(buf + sh_off + off, kTrapPattern, n);
    }
    filled = true;
    break;
  }

  if (!filled) {
    std::free(buf);
    return nullptr;
  }
  if (out_size) *out_size = size;
  return buf;
}

// ---- KernelSwapTable --------------------------------------------------------

void KernelSwapTable::Insert(uint64_t tag_kernel_object,
                             std::shared_ptr<std::vector<uint8_t>> source_object,
                             std::string name) {
  std::scoped_lock lock(mutex_);
  KernelEntry& e = map_[tag_kernel_object];
  e.source_object = std::move(source_object);
  e.name = std::move(name);
  // real_kernel_object stays 0 until transpiled.
}

bool KernelSwapTable::Lookup(uint64_t tag_kernel_object, KernelEntry* out) const {
  std::scoped_lock lock(mutex_);
  const auto it = map_.find(tag_kernel_object);
  if (it == map_.end()) return false;
  if (out) *out = it->second;
  return true;
}

void KernelSwapTable::SetReal(uint64_t tag_kernel_object,
                              uint64_t real_kernel_object,
                              uint32_t private_segment_size,
                              uint32_t group_segment_size) {
  std::scoped_lock lock(mutex_);
  const auto it = map_.find(tag_kernel_object);
  if (it != map_.end()) {
    it->second.real_kernel_object = real_kernel_object;
    it->second.private_segment_size = private_segment_size;
    it->second.group_segment_size = group_segment_size;
  }
}

uint64_t KernelSwapTable::Resolve(uint64_t tag_kernel_object) const {
  std::scoped_lock lock(mutex_);
  const auto it = map_.find(tag_kernel_object);
  if (it == map_.end()) return tag_kernel_object;  // not ours -> pass through
  return it->second.real_kernel_object;            // 0 if not yet transpiled
}

std::vector<std::pair<uint64_t, std::string>> KernelSwapTable::EntriesForSource(
    const void* source) const {
  std::vector<std::pair<uint64_t, std::string>> out;
  std::scoped_lock lock(mutex_);
  for (const auto& kv : map_) {
    if (static_cast<const void*>(kv.second.source_object.get()) == source) {
      out.emplace_back(kv.first, kv.second.name);
    }
  }
  return out;
}

size_t KernelSwapTable::Size() const {
  std::scoped_lock lock(mutex_);
  return map_.size();
}

// ---- AQL packet rewrite -----------------------------------------------------

namespace {
// AQL packet header: bits [0..7] hold the packet type (hsa_packet_type_t).
inline uint8_t PacketType(uint16_t header) {
  return static_cast<uint8_t>((header >> HSA_PACKET_HEADER_TYPE) &
                              ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
}
}  // namespace

size_t RewritePackets(void* packets, uint64_t packet_count,
                      const KernelSwapTable& table) {
  if (!packets || packet_count == 0) return 0;
  auto* pkts = static_cast<hsa_kernel_dispatch_packet_t*>(packets);
  size_t swapped = 0;
  for (uint64_t i = 0; i < packet_count; ++i) {
    hsa_kernel_dispatch_packet_t& p = pkts[i];
    if (PacketType(p.header) != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;
    KernelEntry entry;
    // Unknown handle: pass through. Ours but not yet transpiled
    // (real_kernel_object == 0): leave for the caller to handle.
    if (!table.Lookup(p.kernel_object, &entry)) continue;
    if (entry.real_kernel_object == 0 ||
        entry.real_kernel_object == p.kernel_object)
      continue;
    p.kernel_object = entry.real_kernel_object;
    // HIP built this packet from the tag stub (private/group segment = 0). The
    // real transpiled kernel may need scratch / LDS, so carry its segment sizes
    // or the command processor rejects the dispatch (REGISTER_SIZE_INVALID).
    p.private_segment_size = entry.private_segment_size;
    p.group_segment_size = entry.group_segment_size;
    ++swapped;
  }
  return swapped;
}

// ---- Runtime integration ----------------------------------------------------

bool IsOnDispatchEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("HSA_HOTSWAP_ONDISPATCH");
    return v && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

KernelSwapTable& GlobalSwapTable() {
  static KernelSwapTable* table = new KernelSwapTable();
  return *table;
}

}  // namespace dispatch
}  // namespace hotswap
}  // namespace rocr

#ifdef HSA_RUNTIME_CORE_HOTSWAP_DISPATCH_RUNTIME
#include "inc/hsa_api_trace.h"
#include "core/inc/hotswap.hpp"
#include "core/inc/hotswap_gfx_query.hpp"
#include "core/inc/hsa_internal.h"

namespace rocr {
namespace hotswap {
namespace dispatch {

namespace {

// Verbose diagnostic logging, gated on HSA_HOTSWAP_VERBOSE like the rest of the
// HotSwap code. Off by default.
bool OnDispatchVerbose() {
  static const bool v = [] {
    const char* e = std::getenv("HSA_HOTSWAP_VERBOSE");
    return e && e[0] && e[0] != '0';
  }();
  return v;
}
#define HOTSWAP_ONDISPATCH_LOG(...)                                             \
  do {                                                                         \
    if (OnDispatchVerbose()) fprintf(stderr, __VA_ARGS__);                     \
  } while (false)

// Side executables that hold the real transpiled kernels. Keyed by the source
// object's shared_ptr identity so one transpile+load serves every kernel from
// the same code object. Never destroyed (their kernel_object handles must stay
// live for the process); the retained transpiled ELF buffers live here too.
struct SideExecutable {
  hsa_executable_t executable{0};
  OwnedElfBuffer elf{nullptr, &std::free};
};
std::mutex& side_exec_mutex() {
  static std::mutex* m = new std::mutex();
  return *m;
}
std::unordered_map<const void*, SideExecutable>& side_exec_map() {
  static auto* m = new std::unordered_map<const void*, SideExecutable>();
  return *m;
}

// Transpile `source` for `agent`, load into a fresh frozen executable, and map
// every kernel's tag handle -> real handle in the global table. Returns the
// loaded executable (handle 0 on failure). Loads once per source object.
hsa_executable_t LoadRealExecutableForSource(
    hsa_agent_t agent,
    const std::shared_ptr<std::vector<uint8_t>>& source) {
  const void* key = static_cast<const void*>(source.get());
  {
    std::scoped_lock lock(side_exec_mutex());
    const auto it = side_exec_map().find(key);
    if (it != side_exec_map().end()) return it->second.executable;
  }

  // 1. Transpile the source object down to the device ISA (whole-object; the
  //    COMGR cache makes repeated calls cheap and a later refinement can slice
  //    to a single kernel).
  OwnedElfBuffer rewritten(nullptr, &std::free);
  size_t rewritten_size = 0;
  CodeObjectView view;
  view.data = source->data();
  view.size = source->size();
  // TryRetargetCodeObject now returns a RetargetCodeObjectResult (status +
  // rewrite_required) rather than a bool; a successful transpile reports
  // kRewritten and fills the output buffer.
  RetargetCodeObjectResult retarget =
      TryRetargetCodeObject(view, agent, &rewritten, &rewritten_size);
  if (retarget.status != RetargetCodeObjectStatus::kRewritten ||
      !rewritten || rewritten_size == 0) {
    HOTSWAP_ONDISPATCH_LOG("hotswap[ondispatch]: transpile failed for source %p\n",
                           key);
    return {0};
  }

  // 2. Create + load + freeze a side executable holding the real kernels.
  hsa_executable_t exec{0};
  if (HSA::hsa_executable_create_alt(HSA_PROFILE_FULL,
                                     HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     nullptr, &exec) != HSA_STATUS_SUCCESS) {
    return {0};
  }
  hsa_code_object_reader_t reader{0};
  if (HSA::hsa_code_object_reader_create_from_memory(
          rewritten.get(), rewritten_size, &reader) != HSA_STATUS_SUCCESS) {
    HSA::hsa_executable_destroy(exec);
    return {0};
  }
  hsa_status_t st = HSA::hsa_executable_load_agent_code_object(
      exec, agent, reader, nullptr, nullptr);
  if (st == HSA_STATUS_SUCCESS) {
    st = HSA::hsa_executable_freeze(exec, nullptr);
  }
  HSA::hsa_code_object_reader_destroy(reader);
  if (st != HSA_STATUS_SUCCESS) {
    HSA::hsa_executable_destroy(exec);
    return {0};
  }

  // 3. For every kernel we recorded from this source object, resolve its real
  //    handle by name in the side executable and wire tag -> real.
  KernelSwapTable& table = GlobalSwapTable();
  for (const auto& kv : table.EntriesForSource(source.get())) {
    const uint64_t tag_kobj = kv.first;
    const std::string& name = kv.second;
    hsa_executable_symbol_t sym{0};
    if (HSA::hsa_executable_get_symbol_by_name(exec, name.c_str(), &agent,
                                               &sym) != HSA_STATUS_SUCCESS)
      continue;
    uint64_t real_kobj = 0;
    if (HSA::hsa_executable_symbol_get_info(
            sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &real_kobj) ==
            HSA_STATUS_SUCCESS &&
        real_kobj != 0) {
      // Capture the real kernel's scratch (private) and LDS (group) segment
      // sizes so the dispatch interceptor can fix up the AQL packet HIP built
      // from the zero-segment tag stub.
      uint32_t private_seg = 0;
      uint32_t group_seg = 0;
      HSA::hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
          &private_seg);
      HSA::hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
          &group_seg);
      table.SetReal(tag_kobj, real_kobj, private_seg, group_seg);
    }
  }

  {
    std::scoped_lock lock(side_exec_mutex());
    // Another thread may have populated concurrently; keep the first winner.
    auto& slot = side_exec_map()[key];
    if (slot.executable.handle == 0) {
      slot.executable = exec;
      slot.elf = std::move(rewritten);
    } else {
      HSA::hsa_executable_destroy(exec);
      return slot.executable;
    }
  }
  return exec;
}

}  // namespace

uint64_t EnsureRealKernel(hsa_agent_t agent, uint64_t tag_kernel_object) {
  KernelSwapTable& table = GlobalSwapTable();
  // Already resolved? Return the cached real handle.
  const uint64_t existing = table.Resolve(tag_kernel_object);
  if (existing != 0 && existing != tag_kernel_object) return existing;

  KernelEntry entry;
  if (!table.Lookup(tag_kernel_object, &entry)) {
    return 0;  // not ours; caller passes the packet through untouched
  }
  if (!entry.source_object || entry.source_object->empty()) return 0;

  // Transpile + load the side executable for this source object (once), which
  // wires tag -> real for all of its kernels. Then return this kernel's real
  // handle (0 if transpile/lookup failed -> trap stays -> loud fault).
  LoadRealExecutableForSource(agent, entry.source_object);
  return table.Resolve(tag_kernel_object);
}

namespace {
// Record kernel_object -> {source bytes, name} for every kernel symbol in the
// just-loaded tag executable, keyed by the handle CLR will dispatch.
struct RecordCtx {
  std::shared_ptr<std::vector<uint8_t>> source;
};
hsa_status_t RecordSymbol(hsa_executable_t, hsa_agent_t,
                          hsa_executable_symbol_t sym, void* data) {
  auto* c = static_cast<RecordCtx*>(data);
  hsa_symbol_kind_t kind;
  if (HSA::hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) != HSA_STATUS_SUCCESS ||
      kind != HSA_SYMBOL_KIND_KERNEL)
    return HSA_STATUS_SUCCESS;
  uint64_t kobj = 0;
  if (HSA::hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kobj) !=
          HSA_STATUS_SUCCESS ||
      kobj == 0)
    return HSA_STATUS_SUCCESS;
  uint32_t len = 0;
  if (HSA::hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &len) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  std::string name(len, '\0');
  if (HSA::hsa_executable_symbol_get_info(
          sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()) !=
      HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  GlobalSwapTable().Insert(kobj, c->source, std::move(name));
  return HSA_STATUS_SUCCESS;
}
}  // namespace

bool TryLoadTagForOnDispatch(hsa_executable_t executable, hsa_agent_t agent,
                             const CodeObjectView& code_object,
                             const char* options,
                             hsa_loaded_code_object_t* loaded_code_object,
                             LoadCodeObjectWithSizeFn load_sized, void* context,
                             hsa_status_t* out_status) {
  if (!load_sized || !code_object.data || code_object.size == 0) return false;

  // Only handle gfx1250 source -> supported device target.
  const std::string source_isa =
      GetCodeObjectIsaName(code_object.data, code_object.size);
  const std::string target_isa = GetAgentIsaName(agent);
  const std::string source_gfx = ExtractGfxTarget(source_isa);
  const std::string target_gfx = ExtractGfxTarget(target_isa);
  const uint32_t target_mach = MachForTarget(target_gfx);
  if (source_gfx != "gfx1250" || target_mach == 0) return false;

  // Build the relabel+trap tag from the original bytes.
  size_t tag_size = 0;
  void* tag = MakeRelabeledTrapCopy(code_object.data, code_object.size,
                                    target_mach, &tag_size);
  if (!tag || tag_size == 0) return false;
  OwnedElfBuffer tag_owner(tag, &std::free);

  // Retain the pristine source bytes for per-kernel transpile at dispatch.
  auto source = std::make_shared<std::vector<uint8_t>>(
      static_cast<const uint8_t*>(code_object.data),
      static_cast<const uint8_t*>(code_object.data) + code_object.size);

  // Load the tag so CLR resolves valid handles from it.
  hsa_code_object_t tag_co = {reinterpret_cast<uint64_t>(tag)};
  const hsa_status_t status =
      load_sized(context, agent, tag_co, tag_size, options, code_object.uri,
                 loaded_code_object);
  *out_status = status;
  if (status != HSA_STATUS_SUCCESS) return true;  // handled (and reported)

  // The tag ELF must outlive the executable (loader holds a raw pointer).
  RetainRewrittenElfBuffer(executable, std::move(tag_owner));

  // Record kernel_object -> {source, name} for dispatch-time transpile.
  RecordCtx rc{source};
  HSA::hsa_executable_iterate_agent_symbols(executable, agent, RecordSymbol,
                                            &rc);
  return true;
}

void DispatchInterceptor(const void* packets, uint64_t packet_count,
                         uint64_t /*user_pkt_index*/, void* data,
                         hsa_amd_queue_intercept_packet_writer writer) {
  KernelSwapTable& table = GlobalSwapTable();
  auto* pkts = static_cast<hsa_kernel_dispatch_packet_t*>(
      const_cast<void*>(packets));
  const hsa_agent_t agent = *static_cast<hsa_agent_t*>(data);

  // First-dispatch: for any of our kernels not yet transpiled, do it now
  // (synchronous stall on the cold path).
  for (uint64_t i = 0; i < packet_count; ++i) {
    const uint16_t header = pkts[i].header;
    const uint8_t type = static_cast<uint8_t>(
        (header >> HSA_PACKET_HEADER_TYPE) &
        ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
    if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;
    if (table.Resolve(pkts[i].kernel_object) == 0) {
      EnsureRealKernel(agent, pkts[i].kernel_object);
    }
  }

  RewritePackets(packets ? const_cast<void*>(packets) : nullptr, packet_count,
                 table);
  writer(packets, packet_count);
}

}  // namespace dispatch
}  // namespace hotswap
}  // namespace rocr
#endif  // HSA_RUNTIME_CORE_HOTSWAP_DISPATCH_RUNTIME
