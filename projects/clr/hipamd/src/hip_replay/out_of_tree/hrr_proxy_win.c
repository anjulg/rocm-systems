/* Copyright (c) 2026 Advanced Micro Devices, Inc. - MIT License */

/* HRR Windows proxy DLL.
 *
 * Build: cl /LD /Fe:amdhip64_7.dll hrr_proxy_win.c hrr_trace_writer.c
 *           hrr_code_object.c /link /DEF:amdhip64_7.def
 *
 * Usage: Place amdhip64_7.dll in the application directory alongside
 *        HRR_RECORD=1 environment variable set.
 *
 * The proxy DLL loads the real amdhip64_7.dll via LoadLibraryA using
 * the full path from the HIP installation, then forwards all calls
 * while recording a .hrr trace. */

#ifdef _WIN32

#include <windows.h>
#include <dbghelp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hrr_trace_writer.h"

#define HIP_FAT_MAGIC 0x48495046u  /* "HIPF" */
#define COB_MAGIC "__CLANG_OFFLOAD_BUNDLE__"
#define COB_MAGIC_LEN 24
#define CCOB_MAGIC 0x424f4343u  /* "CCOB" (little-endian read of "CCOB") */

/* ---- CCOB (Compressed Clang Offload Bundle) decompression ----
 *
 * Tensile .co files on recent ROCm builds are CCOB-compressed.  The HIP
 * runtime decompresses them internally during hipModuleLoad, but the proxy
 * needs the decompressed ELF to parse kernel metadata.
 *
 * CCOB header layout (from LLVM OffloadBundle.cpp, all fields LE):
 *   Common (8): Magic(4,"CCOB") + Version(2) + Method(2, 0=zlib 1=zstd)
 *   V1 (20 total): Common + UncompressedSize(4) + Hash(8)
 *   V2 (24 total): Common + TotalFileSize(4) + UncompressedSize(4) + Hash(8)
 *   V3 (32 total): Common + TotalFileSize(8) + UncompressedSize(8) + Hash(8)
 *   Compressed data follows immediately after the header.
 *
 * We dynamically load zstd.dll for decompression (ships with ROCm). */

typedef size_t (__cdecl *pf_ZSTD_decompress)(void*, size_t, const void*, size_t);
typedef unsigned (__cdecl *pf_ZSTD_isError)(size_t);
static pf_ZSTD_decompress pfn_ZSTD_decompress;
static pf_ZSTD_isError pfn_ZSTD_isError;
static int g_zstd_tried = 0;

static int ensure_zstd(void) {
  if (g_zstd_tried) return pfn_ZSTD_decompress != NULL;
  g_zstd_tried = 1;

  HMODULE h = LoadLibraryA("zstd.dll");
  if (!h) h = LoadLibraryA("libzstd.dll");
  if (!h) {
    const char* rocm = getenv("ROCM_PATH");
    if (rocm) {
      char path[MAX_PATH];
      snprintf(path, sizeof(path), "%s\\bin\\zstd.dll", rocm);
      h = LoadLibraryA(path);
      if (!h) {
        snprintf(path, sizeof(path), "%s\\bin\\libzstd.dll", rocm);
        h = LoadLibraryA(path);
      }
    }
  }
  if (!h) {
    fprintf(stderr, "[HRR] zstd.dll not found — cannot decompress CCOB bundles.\n"
            "[HRR] Copy zstd.dll next to the proxy or set ROCM_PATH.\n");
    return 0;
  }
  pfn_ZSTD_decompress = (pf_ZSTD_decompress)GetProcAddress(h, "ZSTD_decompress");
  pfn_ZSTD_isError = (pf_ZSTD_isError)GetProcAddress(h, "ZSTD_isError");
  return pfn_ZSTD_decompress != NULL;
}

/* Decompress a CCOB buffer.  Returns malloc'd buffer on success (caller
 * must free), NULL on failure.  *out_size receives decompressed size. */
static void* hrr_decompress_ccob(const uint8_t* data, size_t data_size,
                                  size_t* out_size) {
  /*
   * CCOB header layout (from LLVM OffloadBundle.cpp):
   *   Common:  Magic(4) + Version(2) + Method(2) = 8 bytes
   *   V1 (20): Common + UncompressedSize(4) + Hash(8)
   *   V2 (24): Common + TotalFileSize(4) + UncompressedSize(4) + Hash(8)
   *   V3 (32): Common + TotalFileSize(8) + UncompressedSize(8) + Hash(8)
   */
  if (data_size < 20) return NULL;

  uint32_t magic;
  memcpy(&magic, data, 4);
  if (magic != CCOB_MAGIC) return NULL;

  uint16_t version, method;
  memcpy(&version, data + 4, 2);
  memcpy(&method, data + 6, 2);

  size_t header_size;
  uint64_t uncompressed_size;
  uint64_t total_file_size = 0;

  switch (version) {
  case 1:
    header_size = 20;
    if (data_size < header_size) return NULL;
    { uint32_t u32; memcpy(&u32, data + 8, 4); uncompressed_size = u32; }
    break;
  case 2:
    header_size = 24;
    if (data_size < header_size) return NULL;
    { uint32_t fs32, us32;
      memcpy(&fs32, data + 8, 4);
      memcpy(&us32, data + 12, 4);
      total_file_size = fs32;
      uncompressed_size = us32;
    }
    break;
  case 3:
    header_size = 32;
    if (data_size < header_size) return NULL;
    memcpy(&total_file_size, data + 8, 8);
    memcpy(&uncompressed_size, data + 16, 8);
    break;
  default:
    fprintf(stderr, "[HRR] CCOB version %u not supported (only 1-3)\n", version);
    return NULL;
  }

  size_t effective_size = data_size;
  if (effective_size > 256u * 1024 * 1024)
    effective_size = 256u * 1024 * 1024;

  size_t compressed_size;
  if (total_file_size > 0 && total_file_size <= effective_size)
    compressed_size = (size_t)(total_file_size - header_size);
  else
    compressed_size = effective_size - header_size;

  if (compressed_size == 0 || uncompressed_size == 0) return NULL;

  if (hrr_verbose())
    fprintf(stderr, "[HRR] CCOB: version=%u method=%u uncompressed=%llu "
            "compressed=%zu header=%zu\n",
            version, method, (unsigned long long)uncompressed_size,
            compressed_size, header_size);

  const uint8_t* compressed_data = data + header_size;

  if (method == 1) {
    if (!ensure_zstd()) return NULL;
    void* buf = malloc((size_t)uncompressed_size);
    if (!buf) return NULL;
    size_t ret = pfn_ZSTD_decompress(buf, (size_t)uncompressed_size,
                                      compressed_data, compressed_size);
    if (pfn_ZSTD_isError && pfn_ZSTD_isError(ret)) {
      fprintf(stderr, "[HRR] CCOB zstd decompression failed\n");
      free(buf);
      return NULL;
    }
    *out_size = (size_t)ret;
    return buf;
  }

  fprintf(stderr, "[HRR] CCOB compression method %u not supported "
          "(only zstd=1 is implemented)\n", method);
  return NULL;
}

/* Parse a decompressed COB buffer: extract GPU ELF code objects and
 * register them via hrr_record_module_load.  Shared by hipModuleLoad
 * and __hipRegisterFatBinary paths. */
static void hrr_process_cob(const uint8_t* data, size_t data_size,
                             void* module_handle) {
  if (data_size < COB_MAGIC_LEN + 8) return;
  if (memcmp(data, COB_MAGIC, COB_MAGIC_LEN) != 0) return;

  uint64_t num_bundles;
  memcpy(&num_bundles, data + COB_MAGIC_LEN, 8);
  if (hrr_verbose())
    fprintf(stderr, "[HRR]   decompressed COB: %llu bundles\n",
            (unsigned long long)num_bundles);

  const uint8_t* hdr = data + COB_MAGIC_LEN + 8;
  for (uint64_t bi = 0; bi < num_bundles && bi < 64; bi++) {
    uint64_t boff, bsz, tlen;
    memcpy(&boff, hdr, 8); hdr += 8;
    memcpy(&bsz,  hdr, 8); hdr += 8;
    memcpy(&tlen, hdr, 8); hdr += 8;

    char triple[256] = {0};
    size_t clen = tlen < sizeof(triple) - 1 ? (size_t)tlen : sizeof(triple) - 1;
    memcpy(triple, hdr, clen);
    hdr += tlen;

    if (hrr_verbose())
      fprintf(stderr, "[HRR]   bundle[%llu]: triple='%s' off=%llu sz=%llu\n",
              (unsigned long long)bi, triple,
              (unsigned long long)boff, (unsigned long long)bsz);

    if (strstr(triple, "amdgcn") && bsz > 4 &&
        boff + bsz <= (uint64_t)data_size) {
      const uint8_t* elf = data + boff;
      if (elf[0] == 0x7f && elf[1] == 'E' && elf[2] == 'L' && elf[3] == 'F') {
        if (hrr_verbose())
          fprintf(stderr, "[HRR]   -> GPU ELF code object (%llu bytes)\n",
                  (unsigned long long)bsz);
        hrr_record_module_load(module_handle, elf, (size_t)bsz);
      }
    }
  }
}

/* ---- Symbol resolution via DbgHelp (kernel name fallback) ----
 *
 * When __hipRegisterFunction is not exported by amdhip64_7.dll (common on
 * some ROCm Windows builds), we fall back to SymFromAddr to resolve the host
 * function pointer passed to hipLaunchKernel into a mangled kernel name.
 * The resolved name is cached in the hrr_register_function table so
 * subsequent launches of the same kernel skip the DbgHelp call. */

#pragma comment(lib, "dbghelp.lib")

static BOOL g_sym_initialized = FALSE;

static void hrr_sym_init(void) {
  if (!g_sym_initialized) {
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    g_sym_initialized = SymInitialize(GetCurrentProcess(), NULL, TRUE);
  }
}

/* Resolve addr → symbol name.  Returns buf on success, NULL on failure. */
static const char* hrr_sym_from_addr(const void* addr, char* buf, DWORD buf_size) {
  if (!g_sym_initialized) return NULL;
  /* SYMBOL_INFO needs trailing space for the name string */
  char sym_storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME];
  SYMBOL_INFO* sym = (SYMBOL_INFO*)sym_storage;
  sym->SizeOfStruct = sizeof(SYMBOL_INFO);
  sym->MaxNameLen   = MAX_SYM_NAME;
  DWORD64 displacement = 0;
  if (SymFromAddr(GetCurrentProcess(), (DWORD64)(uintptr_t)addr,
                  &displacement, sym)) {
    strncpy(buf, sym->Name, (size_t)(buf_size - 1));
    buf[buf_size - 1] = '\0';
    return buf;
  }
  return NULL;
}

/* Real DLL handle */
static HMODULE g_real_dll = NULL;

/* dim3 equivalent for intercepting hipLaunchKernel without C++ headers.
 * On Windows x64 a 12-byte struct is passed by hidden reference; declaring it
 * this way makes the C compiler emit the correct calling convention. */
typedef struct { unsigned int x, y, z; } hrr_dim3;

/* Real function pointers */
typedef int (__cdecl *pf_hipInit)(unsigned int);
typedef int (__cdecl *pf_hipMalloc)(void**, size_t);
typedef int (__cdecl *pf_hipFree)(void*);
typedef int (__cdecl *pf_hipMemcpy)(void*, const void*, size_t, unsigned int);
typedef int (__cdecl *pf_hipMemset)(void*, int, size_t);
typedef int (__cdecl *pf_hipModuleLoad)(void**, const char*);
typedef int (__cdecl *pf_hipModuleLoadData)(void**, const void*);
typedef int (__cdecl *pf_hipModuleUnload)(void*);
typedef int (__cdecl *pf_hipModuleGetFunction)(void**, void*, const char*);
typedef int (__cdecl *pf_hipModuleLaunchKernel)(void*, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, unsigned, void*,
    void**, void**);
typedef int (__cdecl *pf_hipExtModuleLaunchKernel)(void*,
    unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned,
    unsigned long, void*,
    void**, void**, void*, void*, unsigned int);
/* Fat-binary registration — called by DLL global constructors in any library
 * that embeds HIP kernels (e.g. hipblaslt.dll, composable_kernel). */
typedef void** (__cdecl *pf___hipRegisterFatBinary)(const void*);
typedef void (__cdecl *pf___hipRegisterFunction)(
    void**, const void*, char*, const char*,
    int, void*, void*, void*, void*, int*);
/* High-level kernel launch used by fat-binary path (hipLaunchKernelGGL / <<<>>>) */
typedef int (__cdecl *pf_hipLaunchKernel)(
    const void*, hrr_dim3, hrr_dim3, void**, size_t, void*);
/* Returns the registered kernel name for a host function pointer.
 * Uses HIP's own internal fat-binary registry — no PDB files required. */
typedef const char* (__cdecl *pf_hipKernelNameRefByPtr)(const void*, void*);
typedef int (__cdecl *pf_hipDeviceSynchronize)(void);
typedef int (__cdecl *pf_hipStreamSynchronize)(void*);

static pf_hipInit real_hipInit;
static pf_hipMalloc real_hipMalloc;
static pf_hipFree real_hipFree;
static pf_hipMemcpy real_hipMemcpy;
static pf_hipMemset real_hipMemset;
static pf_hipModuleLoad real_hipModuleLoad;
static pf_hipModuleLoadData real_hipModuleLoadData;
static pf_hipModuleUnload real_hipModuleUnload;
static pf_hipModuleGetFunction real_hipModuleGetFunction;
static pf_hipModuleLaunchKernel real_hipModuleLaunchKernel;
static pf_hipExtModuleLaunchKernel real_hipExtModuleLaunchKernel;
static pf___hipRegisterFatBinary real___hipRegisterFatBinary;
static pf___hipRegisterFunction real___hipRegisterFunction;
static pf_hipLaunchKernel real_hipLaunchKernel;
static pf_hipKernelNameRefByPtr real_hipKernelNameRefByPtr;
static pf_hipDeviceSynchronize real_hipDeviceSynchronize;
static pf_hipStreamSynchronize real_hipStreamSynchronize;

static int g_loaded = 0;
static int g_hrr_initialized = 0;

static void ensure_hrr_init(void) {
  if (!g_hrr_initialized) {
    g_hrr_initialized = 1;
    hrr_writer_init();
  }
}

/* Real DLL name after renaming: amdhip64_7.dll -> amdhip64_7_orig.dll */
#define HRR_REAL_DLL_NAME "amdhip64_7_orig.dll"

static void load_real_dll(void) {
  if (g_loaded) return;
  g_loaded = 1;

  /* Search order:
   * 1. HRR_REAL_HIP_PATH env var  — explicit full path to the real DLL
   * 2. Same directory as this proxy DLL  — amdhip64_7_orig.dll beside proxy
   * 3. ROCM_PATH\bin\amdhip64_7_orig.dll
   * 4. System PATH as last resort */

  const char* explicit_path = getenv("HRR_REAL_HIP_PATH");
  if (explicit_path && explicit_path[0]) {
    g_real_dll = LoadLibraryA(explicit_path);
  }

  if (!g_real_dll) {
    /* Resolve the directory that contains this proxy DLL and look for
     * amdhip64_7_orig.dll there.  This is the standard side-by-side layout
     * produced by deploy_proxy.ps1. */
    char self_path[MAX_PATH] = {0};
    HMODULE h_self = NULL;
    GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)load_real_dll, &h_self);
    if (h_self && GetModuleFileNameA(h_self, self_path, sizeof(self_path))) {
      char* last_sep = strrchr(self_path, '\\');
      if (last_sep) {
        strcpy(last_sep + 1, HRR_REAL_DLL_NAME);
        g_real_dll = LoadLibraryA(self_path);
      }
    }
  }

  if (!g_real_dll) {
    const char* rocm_path = getenv("ROCM_PATH");
    if (rocm_path) {
      char buf[MAX_PATH];
      snprintf(buf, sizeof(buf), "%s\\bin\\" HRR_REAL_DLL_NAME, rocm_path);
      g_real_dll = LoadLibraryA(buf);
    }
  }

  if (!g_real_dll) {
    g_real_dll = LoadLibraryA(HRR_REAL_DLL_NAME);
  }

  if (!g_real_dll) {
    fprintf(stderr,
            "[HRR] FATAL: Cannot find real " HRR_REAL_DLL_NAME "\n"
            "[HRR] Run deploy_proxy.ps1 to rename the original DLL, or set\n"
            "[HRR]   HRR_REAL_HIP_PATH=<full path to amdhip64_7_orig.dll>\n");
    ExitProcess(1);
  }

  /* Load function pointers */
#define LOAD(name) real_##name = (pf_##name)GetProcAddress(g_real_dll, #name)
  LOAD(hipInit);
  LOAD(hipMalloc);
  LOAD(hipFree);
  LOAD(hipMemcpy);
  LOAD(hipMemset);
  LOAD(hipModuleLoad);
  LOAD(hipModuleLoadData);
  LOAD(hipModuleUnload);
  LOAD(hipModuleGetFunction);
  LOAD(hipModuleLaunchKernel);
  LOAD(hipExtModuleLaunchKernel);
  LOAD(__hipRegisterFatBinary);
  LOAD(__hipRegisterFunction);
  LOAD(hipLaunchKernel);
  LOAD(hipKernelNameRefByPtr);
  LOAD(hipDeviceSynchronize);
  LOAD(hipStreamSynchronize);
#undef LOAD
}

/* Resolve a kernel function pointer to its name.
 * Priority:
 *   1. hrr_register_function cache  (free, O(n))
 *   2. hipKernelNameRefByPtr        (HIP's own fat-binary registry, no PDB needed)
 *   3. PE export table walk         (works for exported stubs, no PDB needed)
 *   4. SymFromAddr                  (DbgHelp, requires PDB — last resort)
 * On success the resolved name is cached so subsequent launches skip steps 2-4.
 * Set HRR_VERBOSE=1 to print which step resolves (or fails) each pointer. */

static int hrr_verbose(void) {
  static int v = -1;
  if (v < 0) { const char* e = getenv("HRR_VERBOSE"); v = (e && e[0] == '1') ? 1 : 0; }
  return v;
}

/* Compute the true byte extent of an ELF64 binary.
 *
 * The naive formula  e_shoff + e_shentsz * e_shnum  only covers the section
 * HEADER TABLE.  In AMDGPU code objects the section DATA (kernel machine code,
 * symbol table, metadata) is placed AFTER the section header table, so the
 * naive size truncates the binary and loses all kernel code.
 *
 * We fix this by walking every section header and extending the "end" marker
 * to cover each section's data range.
 *
 * max_readable: hard upper bound on bytes safely readable from p.
 *   Pass (size_t)-1 for in-memory blobs (e.g. from hipModuleLoadData) where
 *   the caller guarantees the buffer is valid; the clamp at the end keeps the
 *   returned size within reasonable bounds.
 *   Pass (size_t)fsz for on-disk reads where fsz is the actual file size. */
static size_t elf64_true_size(const unsigned char* p, size_t max_readable) {
  if (max_readable < 64) return 0;
  if (p[0] != 0x7f || p[1] != 'E' || p[2] != 'L' || p[3] != 'F') return 0;
  if (p[4] != 2) return 0;  /* ELFCLASS64 only */

  uint64_t e_shoff;
  uint16_t e_shentsz, e_shnum;
  memcpy(&e_shoff,   p + 40, 8);
  memcpy(&e_shentsz, p + 58, 2);
  memcpy(&e_shnum,   p + 60, 2);

  if (e_shentsz < 64 || e_shnum == 0 || e_shoff == 0) return 0;

  /* Start with the end of the section header table itself */
  uint64_t end = e_shoff + (uint64_t)e_shentsz * e_shnum;

  /* Extend end to cover every section's data range */
  for (uint16_t i = 0; i < e_shnum; i++) {
    uint64_t sh_base = e_shoff + (uint64_t)i * e_shentsz;
    if (sh_base + 64 > (uint64_t)max_readable) break;

    uint32_t sh_type;
    uint64_t sh_offset, sh_size;
    memcpy(&sh_type,   p + sh_base + 4,  4);
    memcpy(&sh_offset, p + sh_base + 24, 8);
    memcpy(&sh_size,   p + sh_base + 32, 8);

    /* SHT_NOBITS (8) has no file representation; also skip null entries */
    if (sh_type == 8 || sh_offset == 0 || sh_size == 0) continue;

    uint64_t section_end = sh_offset + sh_size;
    if (section_end > end) end = section_end;
  }

  if (end > (uint64_t)max_readable) end = (uint64_t)max_readable;
  return (size_t)end;
}

/* Walk the PE export table of the DLL that contains addr. */
static const char* hrr_export_name(const void* addr, char* buf, size_t buf_size) {
  HMODULE hmod = NULL;
  if (!GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)addr, &hmod) || !hmod)
    return NULL;

  BYTE* base = (BYTE*)hmod;
  IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)base;
  if (dos->e_magic != IMAGE_DOS_SIGNATURE) return NULL;
  IMAGE_NT_HEADERS* nt = (IMAGE_NT_HEADERS*)(base + dos->e_lfanew);
  DWORD exp_rva = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress;
  if (!exp_rva) return NULL;

  IMAGE_EXPORT_DIRECTORY* exp = (IMAGE_EXPORT_DIRECTORY*)(base + exp_rva);
  DWORD*  funcs    = (DWORD*)(base + exp->AddressOfFunctions);
  DWORD*  names    = (DWORD*)(base + exp->AddressOfNames);
  WORD*   ordinals = (WORD*) (base + exp->AddressOfNameOrdinals);
  uintptr_t target = (uintptr_t)addr;

  for (DWORD i = 0; i < exp->NumberOfNames; i++) {
    if ((uintptr_t)(base + funcs[ordinals[i]]) == target) {
      const char* name = (const char*)(base + names[i]);
      strncpy(buf, name, buf_size - 1);
      buf[buf_size - 1] = '\0';
      return buf;
    }
  }
  return NULL;
}

static const char* hrr_kernel_name(const void* f, char* buf, size_t buf_size) {
  /* Step 1: cache — populated by __hipRegisterFunction / hipModuleGetFunction */
  const char* kname = hrr_lookup_function_name(f);
  if (kname) {
    if (hrr_verbose())
      fprintf(stderr, "[HRR] cache hit(%p) -> %s\n", f, kname);
    return kname;
  }
  if (hrr_verbose())
    fprintf(stderr, "[HRR] cache miss(%p)\n", f);

  /* Step 2: hipKernelNameRefByPtr — HIP's own fat-binary registry */
  if (real_hipKernelNameRefByPtr) {
    kname = real_hipKernelNameRefByPtr(f, NULL);
    if (hrr_verbose())
      fprintf(stderr, "[HRR] hipKernelNameRefByPtr(%p) -> %s\n",
              f, kname ? kname : "(null)");
    if (kname && kname[0]) {
      hrr_register_function(f, NULL, kname);
      return hrr_lookup_function_name(f);
    }
  } else if (hrr_verbose()) {
    fprintf(stderr, "[HRR] hipKernelNameRefByPtr not available in real DLL\n");
  }

  /* Step 3: PE export table (works without PDB, for exported host stubs).
   * Also prints the containing module name to help diagnose JIT/private stubs. */
  {
    HMODULE hmod_pe = NULL;
    if (GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCSTR)f, &hmod_pe) && hmod_pe) {
      if (hrr_verbose()) {
        char mod_name[MAX_PATH] = {0};
        GetModuleFileNameA(hmod_pe, mod_name, sizeof(mod_name));
        fprintf(stderr, "[HRR] addr %p is in module: %s\n", f, mod_name);
      }
    } else if (hrr_verbose()) {
      fprintf(stderr, "[HRR] addr %p is NOT in any loaded PE module (JIT/heap?)\n", f);
    }
  }
  kname = hrr_export_name(f, buf, buf_size);
  if (hrr_verbose())
    fprintf(stderr, "[HRR] PE export walk(%p) -> %s\n",
            f, kname ? kname : "(null)");
  if (kname) { hrr_register_function(f, NULL, kname); return kname; }

  /* Step 4: DbgHelp SymFromAddr (requires PDB) */
  kname = hrr_sym_from_addr(f, buf, (DWORD)buf_size);
  if (hrr_verbose())
    fprintf(stderr, "[HRR] SymFromAddr(%p) -> %s\n",
            f, kname ? kname : "(null)");
  if (kname) hrr_register_function(f, NULL, kname);
  return kname;
}

/* ---- Exported proxy functions ---- */

__declspec(dllexport) int __cdecl hipInit(unsigned int flags) {
  load_real_dll();
  int ret = real_hipInit(flags);
  ensure_hrr_init();
  return ret;
}

__declspec(dllexport) int __cdecl hipMalloc(void** ptr, size_t size) {
  load_real_dll();
  ensure_hrr_init();
  int ret = real_hipMalloc(ptr, size);
  if (ret == 0 && hrr_writer_enabled()) {
    hrr_record_malloc(*ptr, size, 0);
  }
  return ret;
}

__declspec(dllexport) int __cdecl hipFree(void* ptr) {
  load_real_dll();
  ensure_hrr_init();
  if (hrr_writer_enabled()) hrr_record_free(ptr);
  return real_hipFree(ptr);
}

__declspec(dllexport) int __cdecl hipMemcpy(void* dst, const void* src,
                                            size_t size, unsigned int kind) {
  load_real_dll();
  ensure_hrr_init();
  if (hrr_writer_enabled()) hrr_record_memcpy(dst, src, size, kind, NULL);
  return real_hipMemcpy(dst, src, size, kind);
}

__declspec(dllexport) int __cdecl hipMemset(void* dst, int value, size_t count) {
  load_real_dll();
  ensure_hrr_init();
  if (hrr_writer_enabled()) hrr_record_memset(dst, value, count, NULL);
  return real_hipMemset(dst, value, count);
}

/* hipModuleLoad — loads an HSACO/CO from a file path.
 * hipBLASLt/Tensile uses this to load GEMM kernels from .co/.hsaco files at
 * runtime.  We read the file ourselves to capture the code object for the
 * trace.  Files can be raw ELF or Clang Offload Bundles (COB) — we handle
 * both formats. */
__declspec(dllexport) int __cdecl hipModuleLoad(void** module, const char* fname) {
  load_real_dll();
  ensure_hrr_init();
  if (!real_hipModuleLoad) return -1;
  int ret = real_hipModuleLoad(module, fname);
  if (ret == 0 && module && *module) {
    if (hrr_verbose())
      fprintf(stderr, "[HRR] hipModuleLoad(%s) -> module=%p\n",
              fname ? fname : "(null)", *module);
    if (hrr_writer_enabled() && fname) {
      FILE* f = fopen(fname, "rb");
      if (f) {
        fseek(f, 0, SEEK_END);
        long fsz = ftell(f);
        rewind(f);
        if (fsz > 0) {
          void* buf = malloc((size_t)fsz);
          if (buf && fread(buf, 1, (size_t)fsz, f) == (size_t)fsz) {
            const unsigned char* p = (const unsigned char*)buf;

            if (fsz >= 4 && p[0] == 0x7f && p[1] == 'E' &&
                p[2] == 'L' && p[3] == 'F') {
              /* Raw ELF — capture the full file */
              if (hrr_verbose())
                fprintf(stderr, "[HRR]   format=ELF, size=%ld\n", fsz);
              hrr_record_module_load(*module, buf, (size_t)fsz);

            } else if ((size_t)fsz >= COB_MAGIC_LEN + 8 &&
                       memcmp(p, COB_MAGIC, COB_MAGIC_LEN) == 0) {
              /* Clang Offload Bundle — extract GPU ELF(s).
               * Tensile .co files are often COBs wrapping a single-arch ELF. */
              uint64_t num_bundles;
              memcpy(&num_bundles, p + COB_MAGIC_LEN, 8);
              if (hrr_verbose())
                fprintf(stderr, "[HRR]   format=COB, %llu bundles\n",
                        (unsigned long long)num_bundles);

              const uint8_t* hdr = p + COB_MAGIC_LEN + 8;
              for (uint64_t bi = 0; bi < num_bundles && bi < 64; bi++) {
                uint64_t boff, bsz, tlen;
                memcpy(&boff, hdr, 8); hdr += 8;
                memcpy(&bsz,  hdr, 8); hdr += 8;
                memcpy(&tlen, hdr, 8); hdr += 8;

                char triple[256] = {0};
                size_t clen = tlen < sizeof(triple) - 1
                                  ? (size_t)tlen : sizeof(triple) - 1;
                memcpy(triple, hdr, clen);
                hdr += tlen;

                if (hrr_verbose())
                  fprintf(stderr,
                          "[HRR]   bundle[%llu]: triple='%s' off=%llu sz=%llu\n",
                          (unsigned long long)bi, triple,
                          (unsigned long long)boff, (unsigned long long)bsz);

                if (strstr(triple, "amdgcn") && bsz > 4 &&
                    boff + bsz <= (uint64_t)fsz) {
                  const uint8_t* elf = p + boff;
                  if (elf[0] == 0x7f && elf[1] == 'E' &&
                      elf[2] == 'L' && elf[3] == 'F') {
                    if (hrr_verbose())
                      fprintf(stderr,
                              "[HRR]   -> GPU ELF code object (%llu bytes)\n",
                              (unsigned long long)bsz);
                    hrr_record_module_load(*module, elf, (size_t)bsz);
                  }
                }
              }

            } else if ((size_t)fsz >= 20) {
              uint32_t test_magic;
              memcpy(&test_magic, p, 4);
              if (test_magic == CCOB_MAGIC) {
                /* Compressed Clang Offload Bundle — decompress, then extract
                 * GPU ELF(s) from the inner COB. */
                if (hrr_verbose())
                  fprintf(stderr, "[HRR]   format=CCOB (compressed offload bundle)\n");
                size_t dec_size = 0;
                void* dec = hrr_decompress_ccob(p, (size_t)fsz, &dec_size);
                if (dec && dec_size > 0) {
                  const uint8_t* dp = (const uint8_t*)dec;
                  if (dec_size >= 4 && dp[0] == 0x7f && dp[1] == 'E' &&
                      dp[2] == 'L' && dp[3] == 'F') {
                    hrr_record_module_load(*module, dec, dec_size);
                  } else if (dec_size >= COB_MAGIC_LEN + 8 &&
                             memcmp(dp, COB_MAGIC, COB_MAGIC_LEN) == 0) {
                    hrr_process_cob(dp, dec_size, *module);
                  } else {
                    if (hrr_verbose())
                      fprintf(stderr, "[HRR]   decompressed data not ELF or COB "
                              "(magic=%02x%02x%02x%02x)\n",
                              dp[0], dp[1], dp[2], dp[3]);
                    hrr_record_module_load(*module, dec, dec_size);
                  }
                  free(dec);
                } else {
                  hrr_record_module_load(*module, buf, (size_t)fsz);
                }
              } else {
                /* Unknown format — capture raw bytes as a best-effort fallback. */
                if (hrr_verbose())
                  fprintf(stderr, "[HRR]   format=unknown (magic=%02x%02x%02x%02x), "
                          "capturing raw %ld bytes\n",
                          p[0], p[1], p[2], p[3], fsz);
                hrr_record_module_load(*module, buf, (size_t)fsz);
              }
            }
          }
          free(buf);
        }
        fclose(f);
      }
    }
  }
  return ret;
}

__declspec(dllexport) int __cdecl hipModuleLoadData(void** module,
                                                    const void* image) {
  load_real_dll();
  ensure_hrr_init();
  int ret = real_hipModuleLoadData(module, image);
  if (ret == 0 && hrr_writer_enabled() && module && *module && image) {
    const unsigned char* p = (const unsigned char*)image;
    size_t sz = elf64_true_size(p, (size_t)-1);
    if (sz > 0) {
      hrr_record_module_load(*module, image, sz);
    } else {
      /* Not a raw ELF — check for CCOB (compressed offload bundle) */
      uint32_t test_magic;
      memcpy(&test_magic, p, 4);
      if (test_magic == CCOB_MAGIC) {
        if (hrr_verbose())
          fprintf(stderr, "[HRR] hipModuleLoadData: CCOB detected, decompressing\n");
        /* Buffer size is unknown; pass a large value — hrr_decompress_ccob
         * caps pointer arithmetic internally and ZSTD reads by frame header. */
        size_t dec_size = 0;
        void* dec = hrr_decompress_ccob(p, 128u * 1024 * 1024, &dec_size);
        if (dec && dec_size > 0) {
          const uint8_t* dp = (const uint8_t*)dec;
          if (dec_size >= 4 && dp[0] == 0x7f && dp[1] == 'E' &&
              dp[2] == 'L' && dp[3] == 'F') {
            hrr_record_module_load(*module, dec, dec_size);
          } else if (dec_size >= COB_MAGIC_LEN + 8 &&
                     memcmp(dp, COB_MAGIC, COB_MAGIC_LEN) == 0) {
            hrr_process_cob(dp, dec_size, *module);
          }
          free(dec);
        }
      }
    }
  }
  return ret;
}

__declspec(dllexport) int __cdecl hipModuleUnload(void* module) {
  load_real_dll();
  ensure_hrr_init();
  if (hrr_writer_enabled()) hrr_record_module_unload(module);
  return real_hipModuleUnload(module);
}

/* hipModuleGetFunction — builds the function-handle → (module, kernel-name) map
 * used by kernel launch interception to emit named KERNEL_LAUNCH events.
 * Always registered regardless of HRR_RECORD so the map is ready if recording
 * is enabled mid-process. */
__declspec(dllexport) int __cdecl hipModuleGetFunction(void** hfunc, void* hmod,
                                                       const char* name) {
  load_real_dll();
  ensure_hrr_init();
  if (!real_hipModuleGetFunction) return -1;
  int ret = real_hipModuleGetFunction(hfunc, hmod, name);
  if (ret == 0 && hfunc && *hfunc && name) {
    hrr_register_function(*hfunc, hmod, name);
    if (hrr_verbose())
      fprintf(stderr, "[HRR] hipModuleGetFunction: hfunc=%p mod=%p name='%s'\n",
              *hfunc, hmod, name);
  }
  return ret;
}

/* hipExtModuleLaunchKernel — used by hipBLASLt / composable_kernel on Windows.
 * Uses (globalWorkSize, localWorkSize) semantics; normalized to numBlocks to
 * match the hipModuleLaunchKernel path in the .hrr trace format. */
__declspec(dllexport) int __cdecl hipExtModuleLaunchKernel(
    void* f,
    unsigned int globalWorkSizeX, unsigned int globalWorkSizeY,
    unsigned int globalWorkSizeZ,
    unsigned int localWorkSizeX, unsigned int localWorkSizeY,
    unsigned int localWorkSizeZ,
    unsigned long sharedMemBytes,
    void* hStream,
    void** kernelParams, void** extra,
    void* startEvent, void* stopEvent,
    unsigned int flags) {
  load_real_dll();
  ensure_hrr_init();
  if (hrr_writer_enabled()) {
    char sym_buf[256];
    const char* kname = hrr_kernel_name(f, sym_buf, sizeof(sym_buf));
    uint64_t co_lo = 0, co_hi = 0;
    hrr_lookup_function_co_hash(f, &co_lo, &co_hi);

    /* Convert globalWorkSize → numBlocks (ceiling division) */
    unsigned int nbx = localWorkSizeX ? (globalWorkSizeX + localWorkSizeX - 1) / localWorkSizeX : 1;
    unsigned int nby = localWorkSizeY ? (globalWorkSizeY + localWorkSizeY - 1) / localWorkSizeY : 1;
    unsigned int nbz = localWorkSizeZ ? (globalWorkSizeZ + localWorkSizeZ - 1) / localWorkSizeZ : 1;

    if (kernelParams) {
      hrr_record_kernel_launch(kname, co_lo, co_hi,
                               nbx, nby, nbz,
                               localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                               (uint32_t)sharedMemBytes, hStream, kernelParams);
    } else if (extra) {
      /* Packed kernarg buffer: extra = { HIP_LAUNCH_PARAM_BUFFER_POINTER, buf,
       *                                  HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
       *                                  HIP_LAUNCH_PARAM_END } */
      const void* packed_buf = NULL;
      size_t packed_size = 0;
      int ei;
      for (ei = 0; ; ei += 2) {
        if (extra[ei] == (void*)0x01) {
          packed_buf = extra[ei + 1];
        } else if (extra[ei] == (void*)0x02) {
          packed_size = *(const size_t*)extra[ei + 1];
        } else {
          break;
        }
      }
      if (packed_buf) {
        hrr_record_kernel_launch_packed(kname, co_lo, co_hi,
                                        nbx, nby, nbz,
                                        localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                                        (uint32_t)sharedMemBytes, hStream,
                                        packed_buf, packed_size);
      }
    }
  }
  if (!real_hipExtModuleLaunchKernel) return -1;
  return real_hipExtModuleLaunchKernel(f,
      globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
      localWorkSizeX, localWorkSizeY, localWorkSizeZ,
      sharedMemBytes, hStream,
      kernelParams, extra, startEvent, stopEvent, flags);
}

__declspec(dllexport) int __cdecl hipModuleLaunchKernel(
    void* f, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz,
    unsigned shared, void* stream, void** params, void** extra) {
  load_real_dll();
  ensure_hrr_init();
  if (hrr_writer_enabled()) {
    char sym_buf[256];
    const char* kname = hrr_kernel_name(f, sym_buf, sizeof(sym_buf));
    uint64_t co_lo = 0, co_hi = 0;
    hrr_lookup_function_co_hash(f, &co_lo, &co_hi);
    hrr_record_kernel_launch(kname, co_lo, co_hi,
                             gx, gy, gz, bx, by, bz,
                             shared, stream, params);
  }
  return real_hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz,
                                    shared, stream, params, extra);
}

/* ---- Fat binary code object extraction ----
 *
 * Applications compiled with hipcc embed GPU code objects inside Clang Offload
 * Bundles (COB).  These are registered via __hipRegisterFatBinary at DLL load
 * time rather than through hipModuleLoadData.  We parse the COB to extract and
 * save each GPU ELF code object so the replay can load them. */

/* COB/fat-binary constants are used by both hipModuleLoad and
 * hrr_extract_fat_binary — defined once here. */

static void hrr_extract_fat_binary(const void* fatbin, void* cookie) {
  if (!fatbin) return;

  const uint8_t* data = (const uint8_t*)fatbin;

  /* Check for HIP fat binary wrapper (magic + version + binary ptr + dummy) */
  uint32_t magic;
  memcpy(&magic, data, 4);
  if (magic == HIP_FAT_MAGIC) {
    const void* binary = NULL;
    memcpy(&binary, data + 8, sizeof(void*));
    if (!binary) return;
    data = (const uint8_t*)binary;
    if (hrr_verbose())
      fprintf(stderr, "[HRR] Fat binary wrapper -> binary at %p\n", binary);
  }

  /* Clang Offload Bundle format */
  if (memcmp(data, COB_MAGIC, COB_MAGIC_LEN) == 0) {
    uint64_t num_bundles;
    memcpy(&num_bundles, data + COB_MAGIC_LEN, 8);
    if (hrr_verbose())
      fprintf(stderr, "[HRR] Clang Offload Bundle: %llu bundles\n",
              (unsigned long long)num_bundles);

    const uint8_t* hdr = data + COB_MAGIC_LEN + 8;
    for (uint64_t i = 0; i < num_bundles && i < 64; i++) {
      uint64_t offset, size, triple_len;
      memcpy(&offset, hdr, 8); hdr += 8;
      memcpy(&size,   hdr, 8); hdr += 8;
      memcpy(&triple_len, hdr, 8); hdr += 8;

      char triple[256] = {0};
      size_t copy_len = triple_len < sizeof(triple) - 1
                            ? (size_t)triple_len : sizeof(triple) - 1;
      memcpy(triple, hdr, copy_len);
      hdr += triple_len;

      if (hrr_verbose())
        fprintf(stderr, "[HRR]   bundle[%llu]: triple='%s' offset=%llu size=%llu\n",
                (unsigned long long)i, triple,
                (unsigned long long)offset, (unsigned long long)size);

      /* Only extract GPU bundles (contain "amdgcn") with a valid ELF image */
      if (strstr(triple, "amdgcn") && size > 4) {
        const uint8_t* elf = data + offset;
        if (elf[0] == 0x7f && elf[1] == 'E' && elf[2] == 'L' && elf[3] == 'F') {
          if (hrr_verbose())
            fprintf(stderr, "[HRR]   -> GPU ELF code object (%llu bytes)\n",
                    (unsigned long long)size);
          hrr_record_module_load(cookie, elf, (size_t)size);
        }
      }
    }
    return;
  }

  /* Direct ELF (single-target compilation) */
  if (data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' && data[3] == 'F') {
    size_t sz = elf64_true_size(data, (size_t)-1);
    if (sz > 0) {
      if (hrr_verbose())
        fprintf(stderr, "[HRR] Direct ELF code object (%zu bytes)\n", sz);
      hrr_record_module_load(cookie, data, sz);
    }
  }
}

/* __hipRegisterFatBinary — called by DLL global constructors to register
 * fat binaries containing embedded GPU code objects.  We extract and save each
 * GPU code object so the replay can load them via hipModuleLoadData. */
__declspec(dllexport) void** __cdecl __hipRegisterFatBinary(const void* data) {
  load_real_dll();
  ensure_hrr_init();
  void** cookie = NULL;
  if (real___hipRegisterFatBinary)
    cookie = real___hipRegisterFatBinary(data);
  if (hrr_writer_enabled() && data)
    hrr_extract_fat_binary(data, cookie);
  return cookie;
}

/* __hipRegisterFunction — called by DLL global constructors for each kernel in a
 * fat binary.  Populates the function_ptr → kernel_name map used by hipLaunchKernel
 * so that KERNEL_LAUNCH trace events carry the mangled kernel name.
 * Always runs regardless of HRR_RECORD (the map is always active). */
__declspec(dllexport) void __cdecl __hipRegisterFunction(
    void** modules, const void* hostFunction,
    char* deviceFunction, const char* deviceName,
    int threadLimit, void* tid, void* bid,
    void* blockDim, void* gridDim, int* wSize) {
  load_real_dll();
  if (hrr_verbose())
    fprintf(stderr, "[HRR] __hipRegisterFunction: host=%p device='%s'\n",
            hostFunction, deviceFunction ? deviceFunction : "(null)");
  if (real___hipRegisterFunction)
    real___hipRegisterFunction(modules, hostFunction, deviceFunction, deviceName,
                               threadLimit, tid, bid, blockDim, gridDim, wSize);
  else if (hrr_verbose())
    fprintf(stderr, "[HRR] __hipRegisterFunction: NOT in real DLL — kernel will not launch!\n");
  /* Register even if recording is off — hipLaunchKernel may fire after init. */
  if (hostFunction && deviceFunction)
    hrr_register_function(hostFunction, NULL, deviceFunction);
}

/* hipLaunchKernel — high-level kernel dispatch used by hipBLASLt / composable_kernel
 * fat binaries (the <<<>>> / hipLaunchKernelGGL path). */
__declspec(dllexport) int __cdecl hipLaunchKernel(
    const void* function_address,
    hrr_dim3 numBlocks, hrr_dim3 dimBlocks,
    void** args, size_t sharedMemBytes, void* stream) {
  load_real_dll();
  ensure_hrr_init();
  if (hrr_writer_enabled()) {
    char sym_buf[256];
    const char* kname = hrr_kernel_name(function_address, sym_buf, sizeof(sym_buf));
    uint64_t co_lo = 0, co_hi = 0;
    hrr_lookup_function_co_hash(function_address, &co_lo, &co_hi);
    hrr_record_kernel_launch(kname, co_lo, co_hi,
                             numBlocks.x, numBlocks.y, numBlocks.z,
                             dimBlocks.x, dimBlocks.y, dimBlocks.z,
                             (uint32_t)sharedMemBytes, stream, args);
  }
  if (!real_hipLaunchKernel) return -1;
  return real_hipLaunchKernel(function_address, numBlocks, dimBlocks,
                              args, sharedMemBytes, stream);
}

__declspec(dllexport) int __cdecl hipDeviceSynchronize(void) {
  load_real_dll();
  ensure_hrr_init();
  int ret = real_hipDeviceSynchronize();
  if (hrr_writer_enabled()) hrr_record_device_sync();
  return ret;
}

__declspec(dllexport) int __cdecl hipStreamSynchronize(void* stream) {
  load_real_dll();
  ensure_hrr_init();
  int ret = real_hipStreamSynchronize(stream);
  if (hrr_writer_enabled()) hrr_record_stream_sync(stream);
  return ret;
}

/* DLL entry point */
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved) {
  switch (reason) {
    case DLL_PROCESS_ATTACH:
      /* Initialize the mutex before any other DLL's global constructors run.
       * hipblaslt.dll (and similar libraries) call __hipRegisterFunction from
       * their global constructors, which fires after our DllMain but before
       * the application's main().  hrr_early_init() ensures g.mu is a valid
       * CRITICAL_SECTION before hrr_register_function() tries to lock it. */
      hrr_early_init();
      /* Initialize DbgHelp for SymFromAddr kernel-name fallback. */
      hrr_sym_init();
      break;
    case DLL_PROCESS_DETACH:
      hrr_writer_shutdown();
      if (g_real_dll) FreeLibrary(g_real_dll);
      break;
  }
  return TRUE;
}

#endif /* _WIN32 */
