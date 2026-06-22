/*
 * Shim for a header Apple removed from the macOS 27.0 beta SDK.
 *
 * The Command Line Tools SDK 27.0 still ships CarbonCore's UTCUtils.h, which
 * does `#include <CarbonCore/MacErrors.h>` — but MacErrors.h was deleted from
 * that SDK. UTCUtils.h only guards the include with `#ifndef __MACERRORS__` and
 * uses no symbols from it (just OSStatus/OptionBits from MacTypes.h), so simply
 * defining the include guard satisfies the build. Foundation.h and Accelerate
 * pull UTCUtils.h in transitively, which is why llama.cpp couldn't compile.
 *
 * Put this directory on the compiler's include path (-I .../sdk-shim) so
 * `<CarbonCore/MacErrors.h>` resolves here. Remove once Apple ships a fixed SDK.
 */
#ifndef __MACERRORS__
#define __MACERRORS__
#endif /* __MACERRORS__ */
