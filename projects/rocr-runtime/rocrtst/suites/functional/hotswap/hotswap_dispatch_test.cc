//===- hotswap_dispatch_test.cc - OnDispatch HotSwap unit tests ----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Offline unit tests for the OnDispatch HotSwap subsystem. These exercise the
// pure logic -- ELF relabel+trap, target mach mapping, the kernel swap table,
// and AQL packet rewriting -- with no GPU, no COMGR, and no HSA runtime init.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/inc/hotswap_dispatch.hpp"
#include "gfx1250_min_hsaco.h"
#include "gtest/gtest.h"

namespace {

using rocr::hotswap::dispatch::KernelSwapTable;
using rocr::hotswap::dispatch::MachForTarget;
using rocr::hotswap::dispatch::MakeRelabeledTrapCopy;
using rocr::hotswap::dispatch::RewritePackets;
using rocr::hotswap::dispatch::kEfAmdgpuMachMask;
using rocr::hotswap::dispatch::kElf64EFlagsOffset;
using rocr::hotswap::dispatch::kMachGfx942;
using rocr::hotswap::dispatch::kMachGfx950;

// gfx9 trap pattern the relabel step writes into .text.
constexpr uint8_t kTrapPattern[8] = {0x02, 0x00, 0x92, 0xbf,
                                     0x00, 0x00, 0x81, 0xbf};

// ---- Small ELF64 readers for assertions (independent of the impl) -----------
uint32_t ElfEFlags(const uint8_t* elf) {
  uint32_t v;
  std::memcpy(&v, elf + kElf64EFlagsOffset, sizeof(v));
  return v;
}

// Find a section by name; returns false if absent. Fills file offset + size.
bool FindSection(const uint8_t* elf, size_t size, const char* want,
                 uint64_t* out_off, uint64_t* out_size) {
  uint64_t e_shoff;
  uint16_t e_shentsize, e_shnum, e_shstrndx;
  std::memcpy(&e_shoff, elf + 40, sizeof(e_shoff));
  std::memcpy(&e_shentsize, elf + 58, sizeof(e_shentsize));
  std::memcpy(&e_shnum, elf + 60, sizeof(e_shnum));
  std::memcpy(&e_shstrndx, elf + 62, sizeof(e_shstrndx));
  if (e_shoff + static_cast<uint64_t>(e_shnum) * e_shentsize > size) return false;
  auto f64 = [&](uint16_t i, size_t off) {
    uint64_t v;
    std::memcpy(&v, elf + e_shoff + static_cast<uint64_t>(i) * e_shentsize + off,
                sizeof(v));
    return v;
  };
  auto f32 = [&](uint16_t i, size_t off) {
    uint32_t v;
    std::memcpy(&v, elf + e_shoff + static_cast<uint64_t>(i) * e_shentsize + off,
                sizeof(v));
    return v;
  };
  const uint64_t shstr_off = f64(e_shstrndx, 24);
  for (uint16_t i = 0; i < e_shnum; ++i) {
    const uint32_t name = f32(i, 0);
    const char* nm = reinterpret_cast<const char*>(elf + shstr_off + name);
    if (std::strcmp(nm, want) == 0) {
      if (out_off) *out_off = f64(i, 24);
      if (out_size) *out_size = f64(i, 32);
      return true;
    }
  }
  return false;
}

// Build one AQL kernel-dispatch packet with the given kernel_object.
hsa_kernel_dispatch_packet_t MakeDispatchPacket(uint64_t kernel_object) {
  hsa_kernel_dispatch_packet_t p{};
  p.header = static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH
                                   << HSA_PACKET_HEADER_TYPE);
  p.kernel_object = kernel_object;
  return p;
}

// ---- MachForTarget ----------------------------------------------------------

TEST(HotswapDispatchMach, SupportedTargets) {
  EXPECT_EQ(MachForTarget("gfx942"), kMachGfx942);
  EXPECT_EQ(MachForTarget("gfx950"), kMachGfx950);
}

TEST(HotswapDispatchMach, UnsupportedTargetsReturnZero) {
  EXPECT_EQ(MachForTarget("gfx1250"), 0u);
  EXPECT_EQ(MachForTarget("gfx90a"), 0u);
  EXPECT_EQ(MachForTarget(""), 0u);
}

// ---- MakeRelabeledTrapCopy --------------------------------------------------

TEST(HotswapDispatchRelabel, RejectsMalformedInput) {
  size_t out = 123;
  EXPECT_EQ(MakeRelabeledTrapCopy(nullptr, 0, kMachGfx942, &out), nullptr);
  EXPECT_EQ(out, 0u);
  const uint8_t not_elf[64] = {0};
  EXPECT_EQ(MakeRelabeledTrapCopy(not_elf, sizeof(not_elf), kMachGfx942, &out),
            nullptr);
}

TEST(HotswapDispatchRelabel, RejectsUnsupportedTargetMach) {
  size_t out = 0;
  void* buf =
      MakeRelabeledTrapCopy(kGfx1250MinCo, sizeof(kGfx1250MinCo), 0, &out);
  EXPECT_EQ(buf, nullptr);
  EXPECT_EQ(out, 0u);
}

TEST(HotswapDispatchRelabel, RelabelsMachToTargetGfx942) {
  // Sanity: the fixture is a gfx1250 object (mach 0x49).
  ASSERT_GE(sizeof(kGfx1250MinCo), 64u);
  EXPECT_EQ(ElfEFlags(kGfx1250MinCo) & kEfAmdgpuMachMask, 0x49u);

  size_t out = 0;
  void* buf = MakeRelabeledTrapCopy(kGfx1250MinCo, sizeof(kGfx1250MinCo),
                                    kMachGfx942, &out);
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(out, sizeof(kGfx1250MinCo));
  std::unique_ptr<void, void (*)(void*)> owner(buf, &std::free);
  const uint8_t* elf = static_cast<const uint8_t*>(buf);

  // (a) e_flags mach relabeled; other e_flags bits preserved.
  EXPECT_EQ(ElfEFlags(elf) & kEfAmdgpuMachMask, kMachGfx942 & kEfAmdgpuMachMask);
  EXPECT_EQ(ElfEFlags(elf) & ~kEfAmdgpuMachMask,
            ElfEFlags(kGfx1250MinCo) & ~kEfAmdgpuMachMask);

  // Still a valid ELF64 magic.
  EXPECT_EQ(std::memcmp(elf, "\x7f"
                             "ELF",
                        4),
            0);
}

TEST(HotswapDispatchRelabel, FillsTextWithTrapPreservesRodata) {
  size_t out = 0;
  void* buf = MakeRelabeledTrapCopy(kGfx1250MinCo, sizeof(kGfx1250MinCo),
                                    kMachGfx950, &out);
  ASSERT_NE(buf, nullptr);
  std::unique_ptr<void, void (*)(void*)> owner(buf, &std::free);
  const uint8_t* elf = static_cast<const uint8_t*>(buf);

  // (b) .text is overwritten with the repeating trap pattern.
  uint64_t text_off = 0, text_size = 0;
  ASSERT_TRUE(
      FindSection(elf, sizeof(kGfx1250MinCo), ".text", &text_off, &text_size));
  ASSERT_GT(text_size, 0u);
  for (uint64_t i = 0; i < text_size; ++i) {
    EXPECT_EQ(elf[text_off + i], kTrapPattern[i % sizeof(kTrapPattern)])
        << "mismatch at .text byte " << i;
  }

  // .rodata (kernel descriptors) preserved byte-for-byte if present.
  uint64_t ro_off = 0, ro_size = 0;
  if (FindSection(kGfx1250MinCo, sizeof(kGfx1250MinCo), ".rodata", &ro_off,
                  &ro_size) &&
      ro_size > 0) {
    EXPECT_EQ(std::memcmp(elf + ro_off, kGfx1250MinCo + ro_off, ro_size), 0);
  }
}

// ---- KernelSwapTable --------------------------------------------------------

TEST(HotswapDispatchTable, UnknownHandlePassesThrough) {
  KernelSwapTable t;
  EXPECT_EQ(t.Resolve(0xABCD), 0xABCDu);  // not ours -> return input
  EXPECT_EQ(t.Size(), 0u);
}

TEST(HotswapDispatchTable, KnownButUnresolvedReturnsZero) {
  KernelSwapTable t;
  auto src = std::make_shared<std::vector<uint8_t>>(4, 0u);
  t.Insert(0x1000, src, "kern_a");
  EXPECT_EQ(t.Size(), 1u);
  EXPECT_EQ(t.Resolve(0x1000), 0u);  // ours, not yet transpiled

  rocr::hotswap::dispatch::KernelEntry e;
  ASSERT_TRUE(t.Lookup(0x1000, &e));
  EXPECT_EQ(e.name, "kern_a");
  EXPECT_EQ(e.real_kernel_object, 0u);
  EXPECT_EQ(e.source_object, src);
}

TEST(HotswapDispatchTable, ResolvedReturnsRealHandle) {
  KernelSwapTable t;
  t.Insert(0x1000, std::make_shared<std::vector<uint8_t>>(), "kern_a");
  t.SetReal(0x1000, 0x9999);
  EXPECT_EQ(t.Resolve(0x1000), 0x9999u);
}

TEST(HotswapDispatchTable, EntriesForSourceGroupsByObject) {
  KernelSwapTable t;
  // Two kernels share source object A; one belongs to object B.
  auto a = std::make_shared<std::vector<uint8_t>>(8, 1u);
  auto b = std::make_shared<std::vector<uint8_t>>(8, 2u);
  t.Insert(0x10, a, "a_k0");
  t.Insert(0x11, a, "a_k1");
  t.Insert(0x20, b, "b_k0");

  auto from_a = t.EntriesForSource(a.get());
  auto from_b = t.EntriesForSource(b.get());
  EXPECT_EQ(from_a.size(), 2u);
  EXPECT_EQ(from_b.size(), 1u);
  EXPECT_EQ(from_b[0].first, 0x20u);
  EXPECT_EQ(from_b[0].second, "b_k0");
  // Unknown source -> empty.
  EXPECT_TRUE(t.EntriesForSource(reinterpret_cast<const void*>(0xdead)).empty());
}

// Models the first-dispatch resolve cycle at the table level: a kernel starts
// known-but-unresolved (Resolve==0), then after the side executable is loaded
// (SetReal) the same tag resolves to the real handle and RewritePackets swaps.
TEST(HotswapDispatchTable, FirstDispatchResolveCycle) {
  KernelSwapTable t;
  auto src = std::make_shared<std::vector<uint8_t>>(8, 7u);
  t.Insert(0x500, src, "lazy_kernel");
  EXPECT_EQ(t.Resolve(0x500), 0u);  // cold: not yet transpiled

  // Simulate EnsureRealKernel wiring the real handle after side-load.
  t.SetReal(0x500, 0xCAFE);
  EXPECT_EQ(t.Resolve(0x500), 0xCAFEu);
}

// ---- RewritePackets ---------------------------------------------------------

TEST(HotswapDispatchRewrite, SwapsOnlyResolvedKnownPackets) {
  KernelSwapTable t;
  t.Insert(0x1000, std::make_shared<std::vector<uint8_t>>(), "resolved");
  t.SetReal(0x1000, 0xAAAA);
  t.Insert(0x2000, std::make_shared<std::vector<uint8_t>>(), "pending");
  // 0x2000 deliberately left unresolved (real == 0).

  std::vector<hsa_kernel_dispatch_packet_t> ring = {
      MakeDispatchPacket(0x1000),  // known+resolved -> swap to 0xAAAA
      MakeDispatchPacket(0x2000),  // known+pending  -> leave (0 means "compile")
      MakeDispatchPacket(0x3000),  // unknown        -> pass through
  };

  const size_t swapped = RewritePackets(ring.data(), ring.size(), t);
  EXPECT_EQ(swapped, 1u);
  EXPECT_EQ(ring[0].kernel_object, 0xAAAAu);
  EXPECT_EQ(ring[1].kernel_object, 0x2000u);
  EXPECT_EQ(ring[2].kernel_object, 0x3000u);
}

TEST(HotswapDispatchRewrite, IgnoresNonDispatchPackets) {
  KernelSwapTable t;
  t.Insert(0x1000, std::make_shared<std::vector<uint8_t>>(), "k");
  t.SetReal(0x1000, 0xBBBB);

  // A packet carrying handle 0x1000 but typed as BARRIER_AND, not a dispatch.
  hsa_kernel_dispatch_packet_t barrier = MakeDispatchPacket(0x1000);
  barrier.header = static_cast<uint16_t>(HSA_PACKET_TYPE_BARRIER_AND
                                         << HSA_PACKET_HEADER_TYPE);

  std::vector<hsa_kernel_dispatch_packet_t> ring = {barrier};
  const size_t swapped = RewritePackets(ring.data(), ring.size(), t);
  EXPECT_EQ(swapped, 0u);
  EXPECT_EQ(ring[0].kernel_object, 0x1000u);  // untouched
}

TEST(HotswapDispatchRewrite, HandlesEmptyAndNull) {
  KernelSwapTable t;
  EXPECT_EQ(RewritePackets(nullptr, 4, t), 0u);
  std::vector<hsa_kernel_dispatch_packet_t> ring = {MakeDispatchPacket(0x1)};
  EXPECT_EQ(RewritePackets(ring.data(), 0, t), 0u);
}

}  // namespace
