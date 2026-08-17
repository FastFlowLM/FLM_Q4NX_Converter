/*
 * ff_instr_dump.c — LD_PRELOAD shim to capture FastFlow DPU instruction buffers
 *
 * Intercepts xrt::elf::elf(const char*, size_t) from libxrt_coreutil.so
 * and dumps the ELF payload to disk so the instruction transaction buffer
 * can be extracted from it.
 *
 * Build (host):
 *   gcc -shared -fPIC -O2 \
 *       -I/opt/xilinx/xrt/include \
 *       -o ff_instr_dump.so ff_instr_dump.c -ldl
 *
 * Use:
 *   mkdir -p /tmp/ff_instr
 *   FF_DUMP_INSTR_PATH=/tmp/ff_instr \
 *     LD_PRELOAD=/path/to/ff_instr_dump.so \
 *     /opt/fastflowlm/bin/flm --model GPT-OSS-20B-NPU2 ...
 *
 * Output files:
 *   /tmp/ff_instr/xrt_elf_<N>.elf   — raw ELF buffer passed to xrt::elf()
 *
 * Extract instruction section:
 *   readelf -a /tmp/ff_instr/xrt_elf_0.elf | head -50
 *   # find the section containing the transaction blob, then extract:
 *   objdump -s -j <section_name> /tmp/ff_instr/xrt_elf_0.elf
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * xrt::elf::elf(const char* data, size_t size)
 * mangled: _ZN3xrt3elfC1EPKvm  (and C2 variant)
 *
 * The "this" pointer is the first implicit arg in the Itanium ABI.
 * Signature as seen by the linker: void(xrt::elf*, const char*, size_t)
 */
typedef void (*xrt_elf_ctor_fn)(void* self, const char* data, size_t size);

static pthread_once_t   s_once  = PTHREAD_ONCE_INIT;
static xrt_elf_ctor_fn  s_real  = NULL;
static pthread_mutex_t  s_mutex = PTHREAD_MUTEX_INITIALIZER;
static int              s_count = 0;

static void load_real(void) {
    /* RTLD_NEXT: look for the symbol AFTER our shim in the load order */
    s_real = (xrt_elf_ctor_fn)dlsym(RTLD_NEXT, "_ZN3xrt3elfC1EPKvm");
    if (!s_real) {
        fprintf(stderr,
            "[FF_INSTR_DUMP] RTLD_NEXT: xrt::elf ctor not found: %s\n",
            dlerror());
        /* Fallback: open the library directly */
        void* h = dlopen("libxrt_coreutil.so.2", RTLD_NOW | RTLD_GLOBAL);
        if (!h) h = dlopen("libxrt_coreutil.so", RTLD_NOW | RTLD_GLOBAL);
        if (h) {
            s_real = (xrt_elf_ctor_fn)dlsym(h, "_ZN3xrt3elfC1EPKvm");
        }
    }
    if (!s_real) {
        fprintf(stderr,
            "[FF_INSTR_DUMP] xrt::elf ctor not found via any method — abort\n");
    }
}

/* Called at library load time — confirms the shim is active */
static void __attribute__((constructor)) ff_instr_dump_init(void) {
    const char* dump_dir = getenv("FF_DUMP_INSTR_PATH");
    fprintf(stderr, "[FF_INSTR_DUMP] loaded (FF_DUMP_INSTR_PATH=%s)\n",
            dump_dir ? dump_dir : "(not set)");
}

/* This overrides xrt::elf::elf(const char*, size_t) */
void _ZN3xrt3elfC1EPKvm(void* self, const char* data, size_t size) {
    pthread_once(&s_once, load_real);
    if (!s_real) abort();

    const char* dump_dir = getenv("FF_DUMP_INSTR_PATH");
    if (dump_dir && data && size > 0) {
        pthread_mutex_lock(&s_mutex);
        int idx = s_count++;
        pthread_mutex_unlock(&s_mutex);

        char path[4096];
        snprintf(path, sizeof(path), "%s/xrt_elf_%03d.elf", dump_dir, idx);

        FILE* f = fopen(path, "wb");
        if (f) {
            size_t written = fwrite(data, 1, size, f);
            fclose(f);
            fprintf(stderr,
                "[FF_INSTR_DUMP] xrt::elf #%d  %zu bytes -> %s\n",
                idx, written, path);
        } else {
            fprintf(stderr,
                "[FF_INSTR_DUMP] fopen(%s) failed: %s\n",
                path, strerror(errno));
        }
    }

    s_real(self, data, size);
}

/*
 * Also override the C2 (base ctor) variant — Clang may emit either.
 * Both behave identically for our purposes.
 */
void _ZN3xrt3elfC2EPKvm(void* self, const char* data, size_t size)
    __attribute__((alias("_ZN3xrt3elfC1EPKvm")));

