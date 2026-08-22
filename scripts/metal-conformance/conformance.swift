// A self-verifying Metal conformance battery.
//
// Why this exists. Every result this project has about Maps' missing type layer
// was scored by opening a screenshot, and a screenshot names no seam: it says
// "labels absent" for a wrong pitch, a wrong swizzle, a wrong render-target
// round trip and a dropped draw alike. This binary asks the API directly. Each
// case computes a value the CPU can predict exactly, the GPU produces it, and
// the CPU compares -- so a failure names the case, the expected bytes and the
// bytes that came back.
//
// It is built for two hosts and the comparison between them is the point: run
// it on a native macOS host to establish that a case's expectation is what
// Metal actually does, then run the same binary in the guest. A case that
// passes natively and fails in the guest is a named device defect. A case that
// fails on both is a wrong expectation in this file, not a finding.
//
// Every case reports on one line:  CASE <name> PASS|FAIL <detail>
// and the process exits non-zero if any case failed.

import Metal
import Foundation
import IOSurface

// `MTLCreateSystemDefaultDevice` answers nil in a session with no window
// server attached -- which is every ssh session, and this battery is driven
// over ssh on purpose. `MTLCopyAllDevices` enumerates the same devices without
// that requirement, so the fallback is the contract and not a workaround.
guard let dev = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else {
    print("CASE device FAIL no Metal device")
    exit(2)
}
guard let queue = dev.makeCommandQueue() else {
    print("CASE queue FAIL \(dev.name) would not make a command queue")
    exit(2)
}

var failures = 0
var ran = 0

/// A case the device's own reported limits make inapplicable -- a pitch its
/// `minimumLinearTextureAlignment` forbids, say. Not a failure: Metal would
/// reject the descriptor on any host that reports the same limit, so there is
/// nothing here for a device to get wrong.
var skipped = 0
func skip(_ name: String, _ why: String) {
    skipped += 1
    print("CASE \(name) SKIP \(why)")
    fflush(stdout)
}

func report(_ name: String, _ ok: Bool, _ detail: String) {
    ran += 1
    if !ok { failures += 1 }
    print("CASE \(name) \(ok ? "PASS" : "FAIL") \(detail)")
    fflush(stdout)
}

// ---------------------------------------------------------------------------
// Shaders. Compiled from source at run time on purpose: that exercises the same
// shader path the guest's own apps take, rather than a pre-built archive.
// ---------------------------------------------------------------------------

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

// Exact texel fetch. `read` bypasses the sampler, so a mismatch here is about
// the texture's memory interpretation and nothing else.
kernel void read_texels(texture2d<float, access::read> tex [[texture(0)]],
                        device uint *out [[buffer(0)]],
                        constant uint &width [[buffer(1)]],
                        constant uint2 &extent [[buffer(3)]],
                        device uint *ran [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    // Before the grid guard, so this says the kernel was reached
    // rather than that some thread was in range. Every thread
    // writes the same value; the race is benign and the point is
    // that a dispatch nothing refused cannot leave it zero.
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    float4 v = tex.read(gid);
    uint r = uint(round(v.r * 255.0));
    uint g = uint(round(v.g * 255.0));
    uint b = uint(round(v.b * 255.0));
    uint a = uint(round(v.a * 255.0));
    out[gid.y * width + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}

// The sampler path, with nearest filtering and unnormalized coordinates, so the
// result is still exactly one texel and any difference from `read_texels` is
// the sampler/view rather than the memory.
kernel void sample_texels(texture2d<float, access::sample> tex [[texture(0)]],
                          device uint *out [[buffer(0)]],
                          constant uint &width [[buffer(1)]],
                          constant uint2 &extent [[buffer(3)]],
                          device uint *ran [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    // Before the grid guard, so this says the kernel was reached
    // rather than that some thread was in range. Every thread
    // writes the same value; the race is benign and the point is
    // that a dispatch nothing refused cannot leave it zero.
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_edge);
    float4 v = tex.sample(s, float2(gid.x + 0.5, gid.y + 0.5));
    uint r = uint(round(v.r * 255.0));
    uint g = uint(round(v.g * 255.0));
    uint b = uint(round(v.b * 255.0));
    uint a = uint(round(v.a * 255.0));
    out[gid.y * width + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}

// Read one explicit mip level.
// `coord::pixel` may not be combined with a mip filter, and `level` is the name
// of the LOD constructor, so the uniform is `lod` and the coordinates are
// normalized against this level's own dimensions.
kernel void read_level(texture2d<float, access::sample> tex [[texture(0)]],
                       device uint *out [[buffer(0)]],
                       constant uint &width [[buffer(1)]],
                       constant uint &lod [[buffer(2)]],
                       constant uint2 &extent [[buffer(3)]],
                       device uint *ran [[buffer(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    // Before the grid guard, so this says the kernel was reached
    // rather than that some thread was in range. Every thread
    // writes the same value; the race is benign and the point is
    // that a dispatch nothing refused cannot leave it zero.
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    constexpr sampler s(filter::nearest, mip_filter::nearest, address::clamp_to_edge);
    float2 dim = float2(max(1u, tex.get_width() >> lod), max(1u, tex.get_height() >> lod));
    float2 uv = (float2(gid) + 0.5f) / dim;
    float4 v = tex.sample(s, uv, level(float(lod)));
    uint r = uint(round(v.r * 255.0));
    uint g = uint(round(v.g * 255.0));
    uint b = uint(round(v.b * 255.0));
    uint a = uint(round(v.a * 255.0));
    out[gid.y * width + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}

// The same level, fetched rather than sampled. `read(coord, lod)` names the
// level in the fetch itself, with no sampler and no LOD computation, so a
// device that returns level 0 here has not got the level's *bytes*, while one
// that passes here and fails `read_level` has the bytes and is losing the
// explicit LOD on the sampling path. That is the whole difference between a
// residency bug and a translation bug and nothing else separates them.
kernel void fetch_level(texture2d<float, access::read> tex [[texture(0)]],
                        device uint *out [[buffer(0)]],
                        constant uint &width [[buffer(1)]],
                        constant uint &lod [[buffer(2)]],
                        constant uint2 &extent [[buffer(3)]],
                        device uint *ran [[buffer(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    // Before the grid guard, so this says the kernel was reached
    // rather than that some thread was in range. Every thread
    // writes the same value; the race is benign and the point is
    // that a dispatch nothing refused cannot leave it zero.
    ran[0] = 1u;
    if (gid.x >= extent.x || gid.y >= extent.y) { return; }
    float4 v = tex.read(gid, lod);
    uint r = uint(round(v.r * 255.0));
    uint g = uint(round(v.g * 255.0));
    uint b = uint(round(v.b * 255.0));
    uint a = uint(round(v.a * 255.0));
    out[gid.y * width + gid.x] = (a << 24) | (b << 16) | (g << 8) | r;
}

// Every thread writes to a slot of its own, in a grid padded out to the
// threadgroup size. `dispatchThreads` promises the grid it is given and no
// more, so a slot outside that grid holding a marker is a thread Metal
// promised would not run.
kernel void grid_bounds(device uint *out [[buffer(0)]],
                        constant uint &stride [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * stride + gid.x] = 1u + gid.x;
}

struct VOut { float4 pos [[position]]; float2 uv; };

// A full-target triangle strip driven from a vertex buffer the test owns, so a
// wrong vertex-buffer read shows up as geometry rather than as colour.
vertex VOut quad_vs(uint vid [[vertex_id]],
                    device const float4 *verts [[buffer(0)]]) {
    VOut o;
    float4 v = verts[vid];
    o.pos = float4(v.xy, 0.0, 1.0);
    o.uv = v.zw;
    return o;
}

fragment float4 solid_fs(VOut in [[stage_in]],
                         constant float4 &colour [[buffer(0)]]) {
    return colour;
}

// The same flat colour, made expensive on purpose.
//
// A race is only a test if the arm under test loses it. A solid fill of a
// window-sized target finishes before a host-side reader can decode a copy and
// memcpy three megabytes, so a device that reads those pixels unordered still
// happens to read the right ones and the case passes for a reason that has
// nothing to do with correctness. This shader gives the GPU real per-pixel work
// so the render is still running when the copy behind it is serviced.
//
// The accumulator has to survive the optimizer: `acc` is compared against a
// bound it cannot reach, so the compiler cannot fold the loop away, and the
// colour returned is exactly `colour` on every pixel.
fragment float4 heavy_fs(VOut in [[stage_in]],
                         constant float4 &colour [[buffer(0)]]) {
    float acc = 0.0;
    for (int i = 0; i < 2048; ++i) {
        acc += fract(sin(float(i) * 12.9898 + in.pos.x * 0.017 + in.pos.y * 0.031) * 43758.5453);
    }
    float poison = (acc > 1.0e30) ? 1.0 : 0.0;
    return float4(colour.rgb + poison, colour.a);
}

fragment float4 tex_fs(VOut in [[stage_in]],
                       texture2d<float, access::sample> tex [[texture(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    return tex.sample(s, in.uv);
}
// Coverage type. The atlas carries a single channel, exactly as CoreText
// rasterizes glyphs and exactly the `R8Unorm` a driven Maps boot is observed
// binding, and the colour arrives as a constant. Premultiplied out, so the
// pipeline below can blend it with `one / oneMinusSourceAlpha` the way a text
// layer is composited.
fragment float4 glyph_fs(VOut in [[stage_in]],
                         texture2d<float, access::sample> atlas [[texture(0)]],
                         constant float4 &colour [[buffer(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float cov = atlas.sample(s, in.uv).r;
    return float4(colour.rgb * cov, cov);
}
"""

let library: MTLLibrary
do {
    library = try dev.makeLibrary(source: shaderSource, options: nil)
} catch {
    print("CASE shader_compile FAIL \(error)")
    exit(2)
}
report("shader_compile", true, "runtime library built")

func pipeline(_ name: String) -> MTLComputePipelineState {
    let fn = library.makeFunction(name: name)!
    return try! dev.makeComputePipelineState(function: fn)
}
let readPipe = pipeline("read_texels")
let samplePipe = pipeline("sample_texels")
let levelPipe = pipeline("read_level")
let fetchLevelPipe = pipeline("fetch_level")

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func alignUp(_ v: Int, _ a: Int) -> Int { (v + a - 1) / a * a }

/// Run one of the texel-reading kernels over a texture and return the packed
/// RGBA of every texel, in row-major order.
/// `nil` means the dispatch produced nothing: the kernel never ran.
///
/// **A sentinel fill cannot answer this and must not be asked to.** This device
/// refuses a dispatch on the host side, so the guest's command buffer completes
/// clean and the output buffer keeps whatever was in it. Every case here then
/// compares the sentinel against what it wanted and reports a *content*
/// failure — which reads as "the device returned the wrong bytes" when the
/// truth is that it returned none. The `offset_oracle` cases showed how far
/// that misleads: their fill is `1 + (i % 251)`, zero means "a byte nothing in
/// this buffer ever held", and the sentinel `0xEE` is 238, squarely inside the
/// fill's own range. So a refused dispatch inverted to a constant read-offset
/// of 237, every texel landed in a different delta bucket, and four cases
/// reported `absent=0 shifted=4080` — a precise, plausible account of a defect
/// that did not exist. No sentinel value fixes this in general: a battery whose
/// cases cover many formats has no byte that is out of range for all of them.
///
/// So the kernel says so itself, in a buffer of its own.
func readBack(_ pipe: MTLComputePipelineState,
              _ tex: MTLTexture,
              _ w: Int, _ h: Int,
              level: Int? = nil) -> [UInt32]? {
    let out = dev.makeBuffer(length: w * h * 4, options: .storageModeShared)!
    memset(out.contents(), 0xEE, w * h * 4)
    let ran = dev.makeBuffer(length: 4, options: .storageModeShared)!
    memset(ran.contents(), 0, 4)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setTexture(tex, index: 0)
    enc.setBuffer(out, offset: 0, index: 0)
    var width = UInt32(w)
    enc.setBytes(&width, length: 4, index: 1)
    if var lvl = level.map({ UInt32($0) }) {
        enc.setBytes(&lvl, length: 4, index: 2)
    }
    var extent = SIMD2<UInt32>(UInt32(w), UInt32(h))
    enc.setBytes(&extent, length: 8, index: 3)
    enc.setBuffer(ran, offset: 0, index: 4)
    // Whole threadgroups plus an explicit guard in the kernel, deliberately
    // *not* `dispatchThreads`. The battery's own readback must not depend on
    // the thing `dispatch_threads_grid_*` is here to test: while a device runs
    // the surplus threads of a partial grid, an unguarded readback writes each
    // row's overrun into the next row's first entries and every other case in
    // this file reports that as its own failure.
    let tg = 8
    enc.dispatchThreadgroups(
        MTLSize(width: (w + tg - 1) / tg, height: (h + tg - 1) / tg, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tg, height: tg, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if ran.contents().bindMemory(to: UInt32.self, capacity: 1)[0] == 0 { return nil }
    let p = out.contents().bindMemory(to: UInt32.self, capacity: w * h)
    return Array(UnsafeBufferPointer(start: p, count: w * h))
}

/// A case that cannot be evaluated because the run it depends on did not
/// happen.
///
/// Reported rather than dropped. A battery whose case *count* moves between two
/// runs cannot be diffed against itself, and a name that simply stops appearing
/// reads as a case someone deleted — which is how a refusal of one case quietly
/// took three others out of the totals.
func skipDependent(_ name: String, _ on: String) {
    skip(name, "not evaluated — \(on) never ran")
}

/// One wording for every case, so a refusal is never mistaken for a mismatch.
func refused(_ label: String) {
    report(label, false,
           "the readback dispatch produced nothing — the device refused it, "
           + "or refused a bind in it; the texels below were never written")
}


/// A compact map of where a case's wrong texels are. A count alone cannot tell
/// a lost row from a lost page from scattered noise, and those are three
/// different defects: this prints `y=<row>:<first>-<last>x<count>` per affected
/// row so the shape is in the result line rather than in a follow-up run.
func badMap(_ bad: [(Int, Int)], _ w: Int, _ h: Int) -> String {
    if bad.isEmpty { return "" }
    var perRow: [Int: (Int, Int, Int)] = [:]   // y -> (minX, maxX, count)
    for (x, y) in bad {
        if let e = perRow[y] { perRow[y] = (min(e.0, x), max(e.1, x), e.2 + 1) }
        else { perRow[y] = (x, x, 1) }
    }
    let rows = perRow.keys.sorted()
    let shown = rows.prefix(12).map { y -> String in
        let e = perRow[y]!
        return "y=\(y):\(e.0)-\(e.1)x\(e.2)"
    }.joined(separator: " ")
    let more = rows.count > 12 ? " (+\(rows.count - 12) more rows)" : ""
    return "rows=\(rows.count)/\(h) \(shown)\(more)"
}

func pack(_ r: UInt8, _ g: UInt8, _ b: UInt8, _ a: UInt8) -> UInt32 {
    (UInt32(a) << 24) | (UInt32(b) << 16) | (UInt32(g) << 8) | UInt32(r)
}

func hex(_ v: UInt32) -> String { String(format: "0x%08x", v) }

// ---------------------------------------------------------------------------
// A. A linear texture over a shared buffer, per format, tight and padded pitch.
//
// This is the shape Maps' type layer actually has. The census of a driven boot
// found ~90 distinct `A8Unorm` sources a boot with padded rows -- 54x16 at
// pitch 64, 218x16 at pitch 256, 85x85 at pitch 128 -- so those exact
// geometries are cases here rather than round numbers.
// ---------------------------------------------------------------------------

struct Fmt {
    let name: String
    let mtl: MTLPixelFormat
    let bpp: Int
    /// What Metal must return for a texel whose bytes are `b`.
    let expect: ([UInt8]) -> UInt32
}

let formats: [Fmt] = [
    Fmt(name: "r8Unorm", mtl: .r8Unorm, bpp: 1) { b in pack(b[0], 0, 0, 255) },
    Fmt(name: "a8Unorm", mtl: .a8Unorm, bpp: 1) { b in pack(0, 0, 0, b[0]) },
    Fmt(name: "rg8Unorm", mtl: .rg8Unorm, bpp: 2) { b in pack(b[0], b[1], 0, 255) },
    Fmt(name: "rgba8Unorm", mtl: .rgba8Unorm, bpp: 4) { b in pack(b[0], b[1], b[2], b[3]) },
    Fmt(name: "bgra8Unorm", mtl: .bgra8Unorm, bpp: 4) { b in pack(b[2], b[1], b[0], b[3]) },
]

func linearAliasCase(_ f: Fmt, _ w: Int, _ h: Int, padTo: Int?, sampler: Bool) {
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let tight = alignUp(w * f.bpp, align)
    let bpr = padTo ?? tight
    let label = "linear_\(f.name)_\(w)x\(h)_pitch\(bpr)\(sampler ? "_sampled" : "")"
    if bpr % align != 0 || bpr < w * f.bpp {
        skip(label, "pitch \(bpr) is not a multiple of this device's minimumLinearTextureAlignment=\(align)")
        return
    }
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(label, false, "buffer allocation failed"); return
    }
    // Fill every byte, padding included, with a position-dependent pattern.
    // Padding gets a distinct value so a rail that folds it into the image
    // shows up as a wrong texel rather than as a plausible one.
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    for y in 0..<h {
        for x in 0..<bpr {
            base[y * bpr + x] = UInt8((x &* 7 &+ y &* 13 &+ 11) & 0xFF)
        }
    }
    let d = MTLTextureDescriptor()
    d.textureType = .type2D
    d.pixelFormat = f.mtl
    d.width = w; d.height = h
    d.mipmapLevelCount = 1
    d.storageMode = .shared
    d.usage = sampler ? [.shaderRead] : [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        report(label, false, "makeTexture returned nil"); return
    }
    guard let got = readBack(sampler ? samplePipe : readPipe, tex, w, h) else {
        refused(label); return
    }
    var bad: [(Int, Int)] = []
    var firstDetail = ""
    for y in 0..<h {
        for x in 0..<w {
            var bytes = [UInt8](repeating: 0, count: f.bpp)
            for i in 0..<f.bpp { bytes[i] = base[y * bpr + x * f.bpp + i] }
            let want = f.expect(bytes)
            let have = got[y * w + x]
            if want != have {
                bad.append((x, y))
                if firstDetail.isEmpty {
                    firstDetail = "first_bad=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))"
                }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "\(w * h) texels exact"
                       : "\(bad.count)/\(w * h) wrong \(firstDetail) \(badMap(bad, w, h))")
}

for f in formats {
    linearAliasCase(f, 64, 16, padTo: nil, sampler: false)
    linearAliasCase(f, 64, 16, padTo: nil, sampler: true)
}
// The label-layer geometries, from a driven boot's own census. Their pitches
// are the guest's, so a device whose `minimumLinearTextureAlignment` forbids
// one skips it -- Metal itself would reject the descriptor there, and the
// alignment differs by device (16 on an M-series host, 256 on Apple's
// paravirtual one), so a literal pitch is not portable and must not be a
// failure where it is simply not expressible.
linearAliasCase(formats[1], 54, 16, padTo: 64, sampler: false)
linearAliasCase(formats[1], 54, 16, padTo: 64, sampler: true)
linearAliasCase(formats[1], 218, 16, padTo: 256, sampler: false)
linearAliasCase(formats[1], 85, 85, padTo: 128, sampler: false)
linearAliasCase(formats[0], 128, 16, padTo: 128, sampler: false)
linearAliasCase(formats[4], 60, 8, padTo: 256, sampler: false)

// The same shapes with the padding expressed against whatever this device's
// alignment actually is, so every host runs a padded-pitch case whatever its
// limit. One alignment unit of padding beyond tight, and then two.
for f in formats {
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let tight = alignUp(54 * f.bpp, align)
    for mult in 1...2 {
        linearAliasCase(f, 54, 16, padTo: tight + mult * align, sampler: false)
    }
}

// ---------------------------------------------------------------------------
// B. Incremental CPU writes around GPU reads.
//
// The glyph-atlas pattern: draw with the texture, write more into the unused
// part of the same allocation, draw again, through one texture that is never
// recreated and never re-declared. The contract says the later writes are
// visible; a rail that treats the first read as a snapshot fails here and
// nowhere else.
// ---------------------------------------------------------------------------

/// The glyph-atlas lifecycle, per format and per stage.
///
/// Region A is written before the texture is ever read; region B only after the
/// GPU has already read it once; then A is rewritten. One texture, never
/// recreated, never re-declared. A rail that treats the first read as a
/// snapshot fails here and nowhere else in this file.
///
/// **Run through both stages.** The type layer reaches the rasterizer as a
/// *fragment* texture, and section F exists because a device may route a
/// sampled guest image differently for the two stages. A compute-only version
/// of this case cannot see a defect that lives on the draw path — and on this
/// device `a8Unorm`, the format the type layer uses, is refused on the compute
/// sampled rail outright, so the compute arm of the one format that matters
/// most produces no reading at all. The fragment arm is the one that can.
func incrementalCase(_ f: Fmt, viaFragment: Bool) {
    let w = 64, h = 32, half = 16
    let stage = viaFragment ? "fragment" : "compute"
    let prefix = "incremental_\(f.name)_\(stage)"
    let names = (first: "\(prefix)_first_read",
                 append: "\(prefix)_append_visible",
                 stable: "\(prefix)_untouched_stable",
                 rewrite: "\(prefix)_rewrite_visible")
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = alignUp(w * f.bpp, align)
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(names.first, false, "buffer allocation failed"); return
    }
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    memset(buf.contents(), 0, bpr * h)

    // Byte values chosen so no two phases share one, and none is zero: zero is
    // what an untouched region must read as, so a phase marker that collided
    // with it could not tell "not yet written" from "written and lost".
    let markA: UInt8 = 0x41, markB: UInt8 = 0x5A, markRewrite: UInt8 = 0x77
    func fill(_ rows: Range<Int>, _ value: UInt8) {
        for y in rows { for x in 0..<bpr { base[y * bpr + x] = value } }
    }
    func texel(_ value: UInt8) -> UInt32 { f.expect([UInt8](repeating: value, count: f.bpp)) }
    func read(_ tex: MTLTexture) -> [UInt32]? {
        viaFragment ? fragmentSample(tex, w, h) : readBack(readPipe, tex, w, h)
    }
    func rows(_ got: [UInt32], _ range: Range<Int>, _ want: UInt32) -> Bool {
        range.allSatisfy { y in (0..<w).allSatisfy { x in got[y * w + x] == want } }
    }

    fill(0..<half, markA)
    let d = MTLTextureDescriptor()
    d.textureType = .type2D; d.pixelFormat = f.mtl
    d.width = w; d.height = h; d.mipmapLevelCount = 1
    d.storageMode = .shared; d.usage = [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        report(names.first, false, "makeTexture nil"); return
    }

    guard let first = read(tex) else {
        refused(names.first)
        for dependent in [names.append, names.stable, names.rewrite] {
            skipDependent(dependent, names.first)
        }
        return
    }
    let aOK = rows(first, 0..<half, texel(markA))
    let bZero = rows(first, half..<h, texel(0))
    report(names.first, aOK && bZero,
           aOK && bZero ? "region A present, region B still zero"
                        : "A_ok=\(aOK) B_zero=\(bZero) sample=\(hex(first[0])) "
                          + "want_A=\(hex(texel(markA))) want_B=\(hex(texel(0)))")

    // Region B: written only after the GPU has already read this texture once.
    fill(half..<h, markB)
    guard let second = read(tex) else {
        refused(names.append)
        skipDependent(names.stable, names.append)
        skipDependent(names.rewrite, names.append)
        return
    }
    let bNow = rows(second, half..<h, texel(markB))
    let aStill = rows(second, 0..<half, texel(markA))
    report(names.append, bNow,
           bNow ? "post-read append visible"
                : "append not visible, got=\(hex(second[half * w])) want=\(hex(texel(markB)))")
    report(names.stable, aStill,
           aStill ? "region A unchanged"
                  : "region A moved, got=\(hex(second[0])) want=\(hex(texel(markA)))")

    // And a rewrite of a region already read twice.
    fill(0..<half, markRewrite)
    guard let third = read(tex) else { refused(names.rewrite); return }
    let rewrite = rows(third, 0..<half, texel(markRewrite))
    report(names.rewrite, rewrite,
           rewrite ? "rewrite visible"
                   : "rewrite not visible, got=\(hex(third[0])) want=\(hex(texel(markRewrite)))")
}

for f in formats {
    incrementalCase(f, viaFragment: false)
}
// The fragment arm runs in section F. `texPipeline` is a top-level binding and
// top-level code executes in order, so reading it from here answers nil and
// every fragment case reports a refusal it never made — which is what the
// native oracle caught when this loop ran both arms in place.

// ---------------------------------------------------------------------------
// C/D. Render-target round trips.
//
// Render into a texture, then (C) sample it in a later pass and (D) copy it out
// to a buffer and check the bytes. A device that renders correctly but cannot
// make the result readable again fails exactly one of these, which is the
// distinction a screenshot cannot draw.
// ---------------------------------------------------------------------------

let quadVerts: [Float] = [
    // x, y, u, v -- a triangle strip covering the whole target
    -1, -1, 0, 1,
     1, -1, 1, 1,
    -1,  1, 0, 0,
     1,  1, 1, 0,
]

func makeRenderPipeline(_ fragment: String, _ fmt: MTLPixelFormat) -> MTLRenderPipelineState? {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = library.makeFunction(name: "quad_vs")
    d.fragmentFunction = library.makeFunction(name: fragment)
    d.colorAttachments[0].pixelFormat = fmt
    return try? dev.makeRenderPipelineState(descriptor: d)
}

func renderTargetCases() {
    let w = 64, h = 64
    let fmt = MTLPixelFormat.bgra8Unorm
    guard let pipe = makeRenderPipeline("solid_fs", fmt) else {
        report("rt_pipeline", false, "render pipeline creation failed"); return
    }
    report("rt_pipeline", true, "solid_fs pipeline built")

    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .private
    let rt = dev.makeTexture(descriptor: td)!

    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4, options: .storageModeShared)!
    // 0x40 red, 0x80 green, 0xC0 blue, opaque -- distinct in every channel so a
    // channel-order error is a different failure from a value error.
    var colour: [Float] = [64.0 / 255, 128.0 / 255, 192.0 / 255, 1.0]
    let wantPixel = pack(64, 128, 192, 255)

    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = rt
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    pass.colorAttachments[0].storeAction = .store

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(pipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    enc.setFragmentBytes(&colour, length: 16, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    // C: sample the rendered target in a separate submission.
    guard let sampled = readBack(readPipe, rt, w, h) else {
        refused("rt_render_then_sample"); return
    }
    let allRight = sampled.allSatisfy { $0 == wantPixel }
    report("rt_render_then_sample", allRight,
           allRight ? "\(w * h) texels are the drawn colour"
                    : "want=\(hex(wantPixel)) got=\(hex(sampled[0])) corner=\(hex(sampled[w * h - 1]))")

    // D: copy it out to a buffer and check the bytes the CPU sees.
    let bpr = w * 4
    let out = dev.makeBuffer(length: bpr * h, options: .storageModeShared)!
    memset(out.contents(), 0xEE, bpr * h)
    let cb2 = queue.makeCommandBuffer()!
    let blit = cb2.makeBlitCommandEncoder()!
    blit.copy(from: rt, sourceSlice: 0, sourceLevel: 0,
              sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: w, height: h, depth: 1),
              to: out, destinationOffset: 0,
              destinationBytesPerRow: bpr, destinationBytesPerImage: bpr * h)
    blit.endEncoding()
    cb2.commit()
    cb2.waitUntilCompleted()
    let p = out.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    // bgra8 in memory: B, G, R, A
    let okBytes = (0..<(w * h)).allSatisfy { i in
        p[i * 4] == 192 && p[i * 4 + 1] == 128 && p[i * 4 + 2] == 64 && p[i * 4 + 3] == 255
    }
    report("rt_blit_to_buffer", okBytes,
           okBytes ? "\(w * h) texels exact through blit"
                   : "first=[\(p[0]),\(p[1]),\(p[2]),\(p[3])] expect=[192,128,64,255]")
}
renderTargetCases()

// ---------------------------------------------------------------------------
// E. A mip chain, level by level.
//
// The alias rail refuses a guest mip chain whose per-level offsets and pitches
// the host driver lays out differently, and that refusal is correct. What has
// never been checked is whether the levels the guest declares are *sampled*
// correctly once the copying rail carries them, which is a different question
// and the one a wrong level table would fail.
// ---------------------------------------------------------------------------

func mipCase() {
    let size = 64
    let levels = 7  // 64,32,16,8,4,2,1
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                     width: size, height: size, mipmapped: true)
    d.mipmapLevelCount = levels
    d.storageMode = .shared
    d.usage = [.shaderRead]
    guard let tex = dev.makeTexture(descriptor: d) else {
        report("mip_chain_create", false, "makeTexture nil"); return
    }
    report("mip_chain_create", tex.mipmapLevelCount == levels, "levels=\(tex.mipmapLevelCount)")

    // Each level is filled with a constant that names the level, so a level
    // read at the wrong offset returns a neighbouring level's marker rather
    // than plausible-looking noise.
    for l in 0..<levels {
        let dim = max(1, size >> l)
        let marker = UInt8(0x10 + l * 0x11)
        let rows = [UInt8](repeating: marker, count: dim * dim * 4)
        rows.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake2D(0, 0, dim, dim),
                        mipmapLevel: l,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: dim * 4)
        }
    }
    for l in 0..<levels {
        let dim = max(1, size >> l)
        let marker = UInt8(0x10 + l * 0x11)
        let want = pack(marker, marker, marker, marker)
        // Fetched with the level named in the fetch, then sampled with the
        // level named as an explicit LOD. Which of the two fails says whether
        // the level's bytes are missing or the LOD is.
        guard let fetched = readBack(fetchLevelPipe, tex, dim, dim, level: l) else {
            refused("mip_fetch_level_\(l)_size\(dim)")
            skipDependent("mip_sample_level_\(l)_size\(dim)",
                          "mip_fetch_level_\(l)_size\(dim)")
            continue
        }
        let fOK = fetched.allSatisfy { $0 == want }
        report("mip_fetch_level_\(l)_size\(dim)", fOK,
               fOK ? "\(dim * dim) texels are level \(l)'s marker"
                   : "want=\(hex(want)) got=\(hex(fetched[0]))")
        guard let got = readBack(levelPipe, tex, dim, dim, level: l) else {
            refused("mip_sample_level_\(l)_size\(dim)"); continue
        }
        let ok = got.allSatisfy { $0 == want }
        report("mip_sample_level_\(l)_size\(dim)", ok,
               ok ? "\(dim * dim) texels are level \(l)'s marker"
                  : "want=\(hex(want)) got=\(hex(got[0]))")
    }
}
mipCase()

/// The same chain, uploaded by a blit from a buffer rather than by
/// `replace(region:mipmapLevel:)`. Two different routes reach a level's
/// storage, and a device can lose one and keep the other -- which is the
/// difference between a broken texture-write path and a texture that has no
/// levels above zero at all.
func mipBlitCase() {
    let size = 64
    let levels = 7
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                     width: size, height: size, mipmapped: true)
    d.mipmapLevelCount = levels
    d.storageMode = .private
    d.usage = [.shaderRead, .shaderWrite]
    guard let tex = dev.makeTexture(descriptor: d) else {
        report("mip_blit_create", false, "makeTexture nil"); return
    }
    let cb = queue.makeCommandBuffer()!
    let blit = cb.makeBlitCommandEncoder()!
    var staging: [MTLBuffer] = []
    for l in 0..<levels {
        let dim = max(1, size >> l)
        let marker = UInt8(0x10 + l * 0x11)
        let bpr = dim * 4
        let buf = dev.makeBuffer(length: bpr * dim, options: .storageModeShared)!
        memset(buf.contents(), Int32(marker), bpr * dim)
        staging.append(buf)
        blit.copy(from: buf, sourceOffset: 0,
                  sourceBytesPerRow: bpr, sourceBytesPerImage: bpr * dim,
                  sourceSize: MTLSize(width: dim, height: dim, depth: 1),
                  to: tex, destinationSlice: 0, destinationLevel: l,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    }
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    for l in 0..<levels {
        let dim = max(1, size >> l)
        let marker = UInt8(0x10 + l * 0x11)
        let want = pack(marker, marker, marker, marker)
        guard let got = readBack(fetchLevelPipe, tex, dim, dim, level: l) else {
            refused("mip_blit_level_\(l)_size\(dim)"); continue
        }
        let ok = got.allSatisfy { $0 == want }
        report("mip_blit_level_\(l)_size\(dim)", ok,
               ok ? "\(dim * dim) texels are level \(l)'s marker"
                  : "want=\(hex(want)) got=\(hex(got[0]))")
    }
}
mipBlitCase()

// ---------------------------------------------------------------------------
// F. Vertex-buffer content across submissions.
//
// A draw's geometry comes out of a buffer the guest owns and rewrites between
// frames. If a device caches that window and serves a stale copy, the geometry
// is drawn in last frame's place -- which for a label layer means glyph quads
// landing where nothing is composited, i.e. absence over correct terrain.
// Two submissions with different vertex data, each verified.
// ---------------------------------------------------------------------------

func vertexBufferCase() {
    let w = 64, h = 64
    let fmt = MTLPixelFormat.bgra8Unorm
    guard let pipe = makeRenderPipeline("solid_fs", fmt) else {
        report("vb_pipeline", false, "pipeline failed"); return
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .private
    let rt = dev.makeTexture(descriptor: td)!
    let verts = dev.makeBuffer(length: 4 * 4 * 4, options: .storageModeShared)!
    var colour: [Float] = [1, 1, 1, 1]

    // Draw a strip covering only the requested x half, then check which half of
    // the target changed.
    func drawHalf(_ leftHalf: Bool, _ tag: String) {
        let x0: Float = leftHalf ? -1 : 0
        let x1: Float = leftHalf ? 0 : 1
        let data: [Float] = [x0, -1, 0, 1, x1, -1, 1, 1, x0, 1, 0, 0, x1, 1, 1, 0]
        memcpy(verts.contents(), data, data.count * 4)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = rt
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        enc.setFragmentBytes(&colour, length: 16, index: 0)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        guard let got = readBack(readPipe, rt, w, h) else {
            refused("vb_\(tag)"); return
        }
        let white = pack(255, 255, 255, 255)
        let black = pack(0, 0, 0, 255)
        // Sample well inside each half so rasterization edges are not the test.
        let leftPx = got[(h / 2) * w + (w / 4)]
        let rightPx = got[(h / 2) * w + (3 * w / 4)]
        let want = leftHalf ? (white, black) : (black, white)
        let ok = leftPx == want.0 && rightPx == want.1
        report("vb_\(tag)", ok,
               ok ? "geometry landed in the \(leftHalf ? "left" : "right") half"
                  : "left=\(hex(leftPx)) right=\(hex(rightPx)) wanted left=\(hex(want.0)) right=\(hex(want.1))")
    }
    drawHalf(true, "first_submission_left")
    drawHalf(false, "second_submission_right")
    drawHalf(true, "third_submission_left_again")
}
vertexBufferCase()

// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// F. The same linear alias, bound to a *fragment* stage in a render pass.
//
// Every case above reads the alias from a compute kernel. Type is not drawn
// that way: a glyph atlas reaches the rasterizer as a fragment texture, and a
// device may route a sampled guest image differently for the two stages. If a
// linear alias is exact under `read_texels` and wrong here, the seam is the
// stage binding and not the memory interpretation, which is a distinction no
// screenshot and no compute-only battery can draw.
// ---------------------------------------------------------------------------

/// Draw a full-target quad sampling `tex` into a fresh `rgba8Unorm` target of
/// the same size, and return the target's texels. With nearest filtering and a
/// target sized to the texture, pixel (x, y) samples texel (x, y) exactly.
func fragmentSample(_ tex: MTLTexture, _ w: Int, _ h: Int) -> [UInt32]? {
    guard let pipe = texPipeline else { return nil }
    let rd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    rd.usage = [.renderTarget, .shaderRead]
    rd.storageMode = .private
    guard let rt = dev.makeTexture(descriptor: rd) else { return nil }
    let verts = dev.makeBuffer(bytes: quadVerts,
                               length: quadVerts.count * 4,
                               options: .storageModeShared)!
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = rt
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(pipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    enc.setFragmentTexture(tex, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return readBack(readPipe, rt, w, h)
}

func fragmentAliasCase(_ f: Fmt, _ w: Int, _ h: Int, padTo: Int?) {
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = padTo ?? alignUp(w * f.bpp, align)
    let label = "fragsample_\(f.name)_\(w)x\(h)_pitch\(bpr)"
    if bpr % align != 0 || bpr < w * f.bpp {
        skip(label, "pitch \(bpr) is not a multiple of this device's minimumLinearTextureAlignment=\(align)")
        return
    }
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(label, false, "buffer allocation failed"); return
    }
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    for y in 0..<h { for x in 0..<bpr { base[y * bpr + x] = UInt8((x &* 5 &+ y &* 17 &+ 3) & 0xFF) } }
    let d = MTLTextureDescriptor()
    d.textureType = .type2D; d.pixelFormat = f.mtl
    d.width = w; d.height = h; d.mipmapLevelCount = 1
    d.storageMode = .shared; d.usage = [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        report(label, false, "makeTexture nil"); return
    }

    func check(_ tag: String, _ expectByte: (Int, Int) -> [UInt8]) {
        guard let got = fragmentSample(tex, w, h) else {
            report("\(label)_\(tag)", false, "render pipeline unavailable"); return
        }
        var bad: [(Int, Int)] = []
        var first = ""
        for y in 0..<h {
            for x in 0..<w {
                let want = f.expect(expectByte(x, y))
                let have = got[y * w + x]
                if want != have {
                    bad.append((x, y))
                    if first.isEmpty { first = "first_bad=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))" }
                }
            }
        }
        report("\(label)_\(tag)", bad.isEmpty,
               bad.isEmpty ? "\(w * h) texels exact through the rasterizer"
                           : "\(bad.count)/\(w * h) wrong \(first) \(badMap(bad, w, h))")
    }

    check("first_draw") { x, y in
        (0..<f.bpp).map { base[y * bpr + x * f.bpp + $0] }
    }
    // Rewrite every byte, with the texture already drawn with once and never
    // re-declared. This is the glyph atlas being refilled between frames.
    for y in 0..<h { for x in 0..<bpr { base[y * bpr + x] = UInt8((x &* 3 &+ y &* 29 &+ 200) & 0xFF) } }
    check("after_cpu_rewrite") { x, y in
        (0..<f.bpp).map { base[y * bpr + x * f.bpp + $0] }
    }
}

let texPipeline = makeRenderPipeline("tex_fs", .rgba8Unorm)
if texPipeline == nil {
    report("fragsample_pipeline", false, "tex_fs pipeline would not build")
} else {
    fragmentAliasCase(formats[1], 54, 16, padTo: 64)      // the label geometry
    fragmentAliasCase(formats[1], 218, 16, padTo: 256)
    fragmentAliasCase(formats[1], 85, 85, padTo: 128)
    fragmentAliasCase(formats[3], 64, 16, padTo: nil)     // rgba8, tight
    fragmentAliasCase(formats[4], 60, 8, padTo: 256)      // bgra8, padded
    // And once more with the padding derived from this device's own limit, so
    // a host whose alignment skipped the literal pitches above still runs a
    // padded fragment-sampled alias.
    for f in [formats[1], formats[4]] {
        let a = dev.minimumLinearTextureAlignment(for: f.mtl)
        fragmentAliasCase(f, 54, 16, padTo: alignUp(54 * f.bpp, a) + a)
    }
}

// The glyph-atlas lifecycle from section B, now through the stage the type
// layer actually uses. This is the arm that can see a defect on the draw path,
// and for `a8Unorm` it is the only arm that produces a reading at all.
for f in formats {
    incrementalCase(f, viaFragment: true)
}

// The same lifecycle again, filled by `replaceRegion:` rather than through a
// buffer alias — section H says why the two are different rails. Declared
// there, invoked here, because the fragment arm needs `texPipeline`.
for f in formats {
    replaceRegionCase(f, viaFragment: true)
}


// ---------------------------------------------------------------------------
// F2. The same alias, filled so that every byte names its own offset.
//
// A wrong-pitch read, a wrong-offset read and a byte that simply is not there
// are three different defects and the pattern fills above cannot tell them
// apart -- a mismatch says "not what I wrote" and stops. Here byte `i` holds
// `1 + (i % 251)`, so a returned value inverts to a source offset modulo 251,
// and the *difference* between the offset the contract names and the offset the
// device read is the finding. 251 is the largest prime below 256, so the cycle
// never divides a row pitch and an alias off by a row is not congruent to one
// that is exact. Zero is outside the fill's range entirely, so it means the
// device returned a byte nothing in this buffer ever held.
// ---------------------------------------------------------------------------

func offsetOracleCase(_ w: Int, _ h: Int, padTo: Int?, viaFragment: Bool) {
    let f = formats[1]  // a8Unorm: one byte per texel, so a texel *is* an offset
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = padTo ?? alignUp(w, align)
    let label = "offset_oracle_\(w)x\(h)_pitch\(bpr)\(viaFragment ? "_fragment" : "_compute")"
    if bpr % align != 0 || bpr < w {
        skip(label, "pitch \(bpr) is not a multiple of this device's minimumLinearTextureAlignment=\(align)")
        return
    }
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(label, false, "buffer allocation failed"); return
    }
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    for i in 0..<(bpr * h) { base[i] = UInt8(1 + (i % 251)) }
    let d = MTLTextureDescriptor()
    d.textureType = .type2D; d.pixelFormat = f.mtl
    d.width = w; d.height = h; d.mipmapLevelCount = 1
    d.storageMode = .shared; d.usage = [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        report(label, false, "makeTexture nil"); return
    }
    let got: [UInt32]?
    if viaFragment { got = fragmentSample(tex, w, h) } else { got = readBack(readPipe, tex, w, h) }
    // The reading this whole case exists to make trustworthy: without the
    // kernel's own run witness a refused dispatch arrived here as a buffer full
    // of `0xEE`, which inverts to a valid-looking source offset and reported as
    // `absent=0 shifted=4080`. See `readBack`.
    guard let got else { refused(label); return }

    var absent: [(Int, Int)] = []
    var deltas: [Int: Int] = [:]   // (read offset - contract offset) mod 251 -> count
    for y in 0..<h {
        for x in 0..<w {
            let a = Int((got[y * w + x] >> 24) & 0xFF)
            if a == 0 { absent.append((x, y)); continue }
            let readOffset = (a - 1) % 251
            let wantOffset = (y * bpr + x) % 251
            let delta = ((readOffset - wantOffset) % 251 + 251) % 251
            if delta != 0 { deltas[delta, default: 0] += 1 }
        }
    }
    let ok = absent.isEmpty && deltas.isEmpty
    var detail = "\(w * h) texels at the offsets the contract names"
    if !ok {
        let top = deltas.sorted { $0.value > $1.value }.prefix(4)
            .map { "delta=\($0.key)x\($0.value)" }.joined(separator: " ")
        detail = "absent=\(absent.count) shifted=\(deltas.values.reduce(0, +)) \(top) "
            + badMap(absent, w, h)
    }
    report(label, ok, detail)
}

// Tight and padded, read both ways, so the pitch is the only thing that varies
// between a passing case and a failing one.
for viaFragment in [false, true] {
    let a = dev.minimumLinearTextureAlignment(for: formats[1].mtl)
    offsetOracleCase(a, 16, padTo: a, viaFragment: viaFragment)          // tight for this device
    offsetOracleCase(a - 6, 16, padTo: a, viaFragment: viaFragment)      // padded by 6
    offsetOracleCase(218, 16, padTo: alignUp(218, a), viaFragment: viaFragment)
    offsetOracleCase(54, 16, padTo: alignUp(54, a) + a, viaFragment: viaFragment)
}

// ---------------------------------------------------------------------------
// F3. A render target whose width is not a multiple of eight.
//
// Every failing case above lost exactly `alignUp(w, 8) - w` columns, and the
// two that passed were the two whose width was already a multiple of eight.
// That is a property of the *target*, not of the alias being sampled, so this
// draws a flat colour with no texture bound at all: if it fails the same way,
// nothing about buffer-backed textures is involved and the finding is about
// render-target width.
// ---------------------------------------------------------------------------

func targetWidthCase(_ w: Int, _ h: Int) {
    guard let pipe = solidPipeline else {
        report("rt_width_\(w)", false, "solid pipeline unavailable"); return
    }
    let rd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    rd.usage = [.renderTarget, .shaderRead]
    rd.storageMode = .private
    guard let rt = dev.makeTexture(descriptor: rd) else {
        report("rt_width_\(w)", false, "makeTexture nil"); return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = rt
    pass.colorAttachments[0].loadAction = .clear
    // Cleared to a colour that is neither the drawn colour nor zero, so a
    // failing texel says which of "never drawn" and "never anything" it is.
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(pipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    var colour = SIMD4<Float>(0, 1, 0, 1)
    enc.setFragmentBytes(&colour, length: 16, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    guard let got = readBack(readPipe, rt, w, h) else {
        refused("rt_width_\(w)"); return
    }
    let want = pack(0, 255, 0, 255)
    let clear = pack(255, 0, 255, 255)
    var bad: [(Int, Int)] = []
    var zero = 0, cleared = 0
    for y in 0..<h {
        for x in 0..<w where got[y * w + x] != want {
            bad.append((x, y))
            if got[y * w + x] == 0 { zero += 1 }
            if got[y * w + x] == clear { cleared += 1 }
        }
    }
    let pad = alignUp(w, 8) - w
    report("rt_width_\(w)", bad.isEmpty,
           bad.isEmpty ? "\(w * h) texels are the drawn colour"
                       : "\(bad.count)/\(w * h) wrong zero=\(zero) still_clear=\(cleared) "
                         + "alignUp(w,8)-w=\(pad) \(badMap(bad, w, h))")
}

let solidPipeline = makeRenderPipeline("solid_fs", .rgba8Unorm)
for w in [256, 250, 218, 64, 60, 54, 63, 57] { targetWidthCase(w, 16) }

// ---------------------------------------------------------------------------
// F4. `dispatchThreads` must run the grid it is given and not one thread more.
//
// `dispatchThreads:threadsPerThreadgroup:` takes a thread count, not a
// threadgroup count, and Metal is required to launch exactly that many threads
// however badly the count divides the threadgroup. A device that rounds the
// grid up to whole threadgroups instead runs extra threads with a
// `thread_position_in_grid` outside the grid, and every one of them is a stray
// write at whatever address the shader computes from it.
//
// Nothing about this is visible in a shader that is careful, which is why it
// hides: the damage lands in the *caller's* buffer, at the addresses just past
// each row, and those are the addresses the next row occupies. That is exactly
// the shape `rt_width_*` reports -- the first `alignUp(w, 8) - w` texels of a
// row, zeroed at random, with row zero always intact because nothing is
// dispatched before it.
//
// Here each thread owns a slot in a grid padded to the threadgroup size, so a
// stray thread cannot overwrite a real one and the evidence survives.
// ---------------------------------------------------------------------------

func gridBoundsCase(_ w: Int, _ h: Int, _ tg: Int) {
    let label = "dispatch_threads_grid_\(w)x\(h)_tg\(tg)"
    let pipe = pipeline("grid_bounds")
    let stride = alignUp(w, tg)
    let rows = alignUp(h, tg)
    let out = dev.makeBuffer(length: stride * rows * 4, options: .storageModeShared)!
    memset(out.contents(), 0, stride * rows * 4)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(out, offset: 0, index: 0)
    var strideU = UInt32(stride)
    enc.setBytes(&strideU, length: 4, index: 1)
    enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tg, height: tg, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let p = out.contents().bindMemory(to: UInt32.self, capacity: stride * rows)
    var missing = 0        // inside the grid and never written
    var strayX = 0         // x >= w
    var strayY = 0         // y >= h
    var firstStray = ""
    for y in 0..<rows {
        for x in 0..<stride {
            let v = p[y * stride + x]
            let inGrid = x < w && y < h
            if inGrid {
                if v != UInt32(1 + x) { missing += 1 }
            } else if v != 0 {
                if x >= w { strayX += 1 } else { strayY += 1 }
                if firstStray.isEmpty { firstStray = "first_stray=(\(x),\(y))=\(v)" }
            }
        }
    }
    let ok = missing == 0 && strayX == 0 && strayY == 0
    report(label, ok,
           ok ? "\(w * h) threads ran and nothing outside the grid did"
              : "missing=\(missing) stray_past_width=\(strayX) stray_past_height=\(strayY) \(firstStray)")
}

// A width that divides the threadgroup and one that does not, in both axes.
gridBoundsCase(64, 16, 8)
gridBoundsCase(218, 16, 8)
gridBoundsCase(54, 15, 8)
gridBoundsCase(57, 9, 8)
gridBoundsCase(31, 31, 16)

// ---------------------------------------------------------------------------
// F5. A CPU write into a texture the GPU has already rendered into.
//
// This is the shape of a compositor that draws its geometry on the GPU and
// rasterizes its type on the CPU, into one shared texture. Metal's contract is
// that both writers land: the render pass owns what it drew, the CPU owns what
// it wrote afterwards, and a later read sees each in the region it wrote.
//
// A device that keeps its own copy of the target and writes that copy back into
// the guest's pages after the fact destroys the second writer's bytes and
// nothing reports it. The failure is content, and content is what no counter in
// this project measures -- so it is asked here directly, and asked in both
// orders, because a writeback landing late and a writeback landing early are
// different bugs with the same symptom.
// ---------------------------------------------------------------------------

func cpuWriteAfterRenderCase(_ w: Int, _ h: Int, secondPass: Bool) {
    let label = "cpu_write_after_render_\(w)x\(h)\(secondPass ? "_then_second_pass" : "")"
    guard let pipe = solidPipeline else { report(label, false, "solid pipeline unavailable"); return }
    let rd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    rd.usage = [.renderTarget, .shaderRead]
    // Shared, so the CPU can write it and the GPU can render into it -- which
    // is the whole point and is what a compositor's own surfaces are.
    rd.storageMode = .shared
    guard let rt = dev.makeTexture(descriptor: rd) else {
        report(label, false, "makeTexture nil"); return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func draw(_ colour: SIMD4<Float>, load: Bool) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = rt
        pass.colorAttachments[0].loadAction = load ? .load : .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        // Only the top half, so the bottom half is the CPU's alone and a
        // whole-target writeback is the only thing that could reach it.
        enc.setScissorRect(MTLScissorRect(x: 0, y: 0, width: w, height: h / 2))
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // 1. The GPU renders into the top half.
    draw(SIMD4<Float>(0, 1, 0, 1), load: false)

    // 2. The CPU rasterizes into the bottom half, after the pass completed.
    let bottom = h / 2
    let bpr = w * 4
    var rows = [UInt8](repeating: 0, count: bpr * (h - bottom))
    for y in 0..<(h - bottom) {
        for x in 0..<w {
            let o = y * bpr + x * 4
            rows[o] = UInt8((x &* 7 &+ y &* 3 &+ 1) & 0xFF)
            rows[o + 1] = 0x20
            rows[o + 2] = 0x40
            rows[o + 3] = 0xFF
        }
    }
    rows.withUnsafeBytes { raw in
        rt.replace(region: MTLRegionMake2D(0, bottom, w, h - bottom),
                   mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: bpr)
    }

    // 3. Optionally another pass over the top half only. A device that reloads
    //    its own stale copy for `.load` and stores the whole target back is
    //    caught here and not by the first arm.
    if secondPass { draw(SIMD4<Float>(0, 0, 1, 1), load: true) }

    guard let got = readBack(readPipe, rt, w, h) else {
        refused(label); return
    }
    let drawn = secondPass ? pack(0, 0, 255, 255) : pack(0, 255, 0, 255)
    var topBad: [(Int, Int)] = []
    var cpuBad: [(Int, Int)] = []
    var cpuFirst = ""
    for y in 0..<h {
        for x in 0..<w {
            let have = got[y * w + x]
            if y < bottom {
                if have != drawn { topBad.append((x, y)) }
            } else {
                let o = (y - bottom) * bpr + x * 4
                let want = pack(rows[o], rows[o + 1], rows[o + 2], rows[o + 3])
                if have != want {
                    cpuBad.append((x, y))
                    if cpuFirst.isEmpty {
                        cpuFirst = "at=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))"
                    }
                }
            }
        }
    }
    let ok = topBad.isEmpty && cpuBad.isEmpty
    report(label, ok,
           ok ? "the GPU kept its half and the CPU kept its half"
              : "gpu_half_wrong=\(topBad.count) cpu_half_wrong=\(cpuBad.count) \(cpuFirst) "
                + badMap(cpuBad, w, h))
}

cpuWriteAfterRenderCase(64, 32, secondPass: false)
cpuWriteAfterRenderCase(64, 32, secondPass: true)
cpuWriteAfterRenderCase(256, 64, secondPass: false)
cpuWriteAfterRenderCase(256, 64, secondPass: true)

// ---------------------------------------------------------------------------
// G. A linear alias at a non-zero offset into its allocation.
//
// A glyph atlas is a sub-range of a larger buffer, so the offset is part of the
// contract. A device that resolves the allocation but drops the offset reads
// the right pages and the wrong bytes, which looks exactly like corruption.
// ---------------------------------------------------------------------------

func offsetAliasCase(_ f: Fmt, _ w: Int, _ h: Int, tiles: Int) {
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = alignUp(w * f.bpp, align)
    let stride = alignUp(bpr * h, max(align, 256))
    let label = "offset_alias_\(f.name)_\(w)x\(h)_x\(tiles)"
    guard let buf = dev.makeBuffer(length: stride * tiles, options: .storageModeShared) else {
        report(label, false, "buffer allocation failed"); return
    }
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: stride * tiles)
    for i in 0..<(stride * tiles) { base[i] = UInt8((i &* 31 &+ 7) & 0xFF) }

    var bad = 0
    var first = ""
    for t in 0..<tiles {
        let off = stride * t
        let d = MTLTextureDescriptor()
        d.textureType = .type2D; d.pixelFormat = f.mtl
        d.width = w; d.height = h; d.mipmapLevelCount = 1
        d.storageMode = .shared; d.usage = [.shaderRead]
        guard let tex = buf.makeTexture(descriptor: d, offset: off, bytesPerRow: bpr) else {
            report(label, false, "makeTexture nil at offset \(off)"); return
        }
        guard let got = readBack(readPipe, tex, w, h) else {
            refused(label); return
        }
        for y in 0..<h {
            for x in 0..<w {
                let bytes = (0..<f.bpp).map { base[off + y * bpr + x * f.bpp + $0] }
                let want = f.expect(bytes)
                if got[y * w + x] != want {
                    bad += 1
                    if first.isEmpty {
                        first = "tile=\(t) offset=\(off) at=(\(x),\(y)) want=\(hex(want)) got=\(hex(got[y * w + x]))"
                    }
                }
            }
        }
    }
    report(label, bad == 0,
           bad == 0 ? "\(tiles) tiles exact at their own offsets" : "\(bad) wrong \(first)")
}

offsetAliasCase(formats[1], 54, 16, tiles: 4)
offsetAliasCase(formats[3], 32, 32, tiles: 3)

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// H. `replaceRegion:` into a sampled texture, sub-rect, across submissions.
//
// Section B writes an atlas through the `MTLBuffer` its texture aliases. That
// is one of the two ways a glyph atlas is filled and it is not the one a
// CPU-rasterized layer uses: CoreGraphics rasterizes into its own bitmap and
// the result reaches Metal through
// `-[MTLTexture replaceRegion:mipmapLevel:withBytes:bytesPerRow:]`, on a plain
// texture with no buffer behind it. That is a different record on the wire and
// a different rail in the device, and nothing in this file reached it except
// inside the mip cases, where a separate defect masks the result.
//
// The sub-rect is the point. An atlas grows by replacing the strip it just
// rasterized, leaving every glyph already in it untouched, so a rail that
// re-uploads the whole texture, or uploads the strip to the wrong origin, or
// drops the second replace because it already has the resource resident, loses
// exactly the glyphs that were there before — which is type that renders once
// and then does not.
// ---------------------------------------------------------------------------

func replaceRegionCase(_ f: Fmt, viaFragment: Bool) {
    let w = 64, h = 32, half = 16
    let stage = viaFragment ? "fragment" : "compute"
    let prefix = "replace_\(f.name)_\(stage)"
    let names = (first: "\(prefix)_first_read",
                 append: "\(prefix)_append_visible",
                 stable: "\(prefix)_untouched_stable",
                 rewrite: "\(prefix)_rewrite_visible")

    let d = MTLTextureDescriptor()
    d.textureType = .type2D; d.pixelFormat = f.mtl
    d.width = w; d.height = h; d.mipmapLevelCount = 1
    // Not buffer-backed, and not `.private`: `replaceRegion` is a CPU write and
    // needs storage the CPU can reach. This is what a rasterized atlas is.
    d.storageMode = .shared; d.usage = [.shaderRead]
    guard let tex = dev.makeTexture(descriptor: d) else {
        report(names.first, false, "makeTexture nil"); return
    }

    let markA: UInt8 = 0x41, markB: UInt8 = 0x5A, markRewrite: UInt8 = 0x77
    // Every texel of the strip, tight — `replaceRegion` takes the caller's own
    // pitch and has no alignment rule of its own.
    func replace(_ y0: Int, _ rows: Int, _ value: UInt8) {
        let pitch = w * f.bpp
        let bytes = [UInt8](repeating: value, count: pitch * rows)
        bytes.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake2D(0, y0, w, rows),
                        mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: pitch)
        }
    }
    func texel(_ value: UInt8) -> UInt32 { f.expect([UInt8](repeating: value, count: f.bpp)) }
    func read() -> [UInt32]? {
        viaFragment ? fragmentSample(tex, w, h) : readBack(readPipe, tex, w, h)
    }
    func rows(_ got: [UInt32], _ range: Range<Int>, _ want: UInt32) -> Bool {
        range.allSatisfy { y in (0..<w).allSatisfy { x in got[y * w + x] == want } }
    }
    // Where a wrong sub-rect landed, which a count alone cannot say.
    func wrongRows(_ got: [UInt32], _ range: Range<Int>, _ want: UInt32) -> String {
        let bad = range.filter { y in !(0..<w).allSatisfy { x in got[y * w + x] == want } }
        guard let firstBad = bad.first else { return "" }
        return " wrong_rows=\(bad.count)/\(range.count) first=\(firstBad) " +
               "got=\(hex(got[firstBad * w])) want=\(hex(want))"
    }

    // A texture from `makeTexture` has undefined contents, so region B is
    // written once here to establish a known floor. Only the *later* writes are
    // the thing under test.
    replace(0, h, 0)
    replace(0, half, markA)
    guard let first = read() else {
        refused(names.first)
        for dependent in [names.append, names.stable, names.rewrite] {
            skipDependent(dependent, names.first)
        }
        return
    }
    let aOK = rows(first, 0..<half, texel(markA))
    let bZero = rows(first, half..<h, texel(0))
    report(names.first, aOK && bZero,
           aOK && bZero ? "the replaced strip is present and the rest is still zero"
                        : "A_ok=\(aOK) B_zero=\(bZero)"
                          + wrongRows(first, 0..<half, texel(markA))
                          + wrongRows(first, half..<h, texel(0)))

    // The atlas grows: a strip replaced after the GPU has already sampled it.
    replace(half, h - half, markB)
    guard let second = read() else {
        refused(names.append)
        skipDependent(names.stable, names.append)
        skipDependent(names.rewrite, names.append)
        return
    }
    report(names.append, rows(second, half..<h, texel(markB)),
           rows(second, half..<h, texel(markB))
             ? "a strip replaced after a read is visible"
             : "append not visible" + wrongRows(second, half..<h, texel(markB)))
    report(names.stable, rows(second, 0..<half, texel(markA)),
           rows(second, 0..<half, texel(markA))
             ? "the glyphs already in the atlas are untouched"
             : "the earlier strip moved" + wrongRows(second, 0..<half, texel(markA)))

    // And a strip rewritten in place, which is how an atlas recycles a slot.
    replace(0, half, markRewrite)
    guard let third = read() else { refused(names.rewrite); return }
    report(names.rewrite, rows(third, 0..<half, texel(markRewrite)),
           rows(third, 0..<half, texel(markRewrite))
             ? "a strip rewritten in place is visible"
             : "rewrite not visible" + wrongRows(third, 0..<half, texel(markRewrite)))
}

for f in formats {
    replaceRegionCase(f, viaFragment: false)
}


// ---------------------------------------------------------------------------
// I. Render targets the guest allocated in its own memory.
//
// Every render target above this line is `.private`, so the device allocates it
// and the guest never names the pages behind it. That is one of the two kinds
// of render target a compositing app uses and it is not the interesting one: a
// layer that the CPU also rasterizes into, or that another process composites,
// is `.shared`, and its bytes are guest memory the device may bind a Vulkan
// image directly over instead of copying.
//
// Those are two different rails with two different failure modes, and until
// this section the battery had exactly one case on the second of them
// (`cpu_write_after_render`, section F5). A whole rail behind one case is not
// coverage -- it is a single sample that happens to pass, which is what let a
// live defect sit under 173 green cases.
//
// The four questions below are the ones a direct binding over guest pages can
// answer differently from a copy, in the order they get harder:
//
//   draw      -- does rendering into guest-owned pages land at all,
//   load      -- does a second pass see what the first one left, or does the
//                seed for `.load` overwrite it with a stale copy,
//   scissor   -- does a partial write land on the right rows, which is where a
//                row pitch the guest did not agree to shows up as a shear,
//   sample    -- does a later pass sampling that same texture read what was
//                rendered, which is the crossover between "this is a target"
//                and "this is a source" over one allocation.
//
// The widths are chosen so the row pitch cannot be assumed: 60 and 1000 texels
// are 240 and 4000 bytes, neither a multiple of the 256-byte linear alignment
// this device reports, so any rail that confuses the guest's stride with a
// padded one puts the pixels somewhere this section can see.
// ---------------------------------------------------------------------------

func sharedRenderTarget(_ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
    let rd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    rd.usage = [.renderTarget, .shaderRead]
    rd.storageMode = .shared
    guard let rt = dev.makeTexture(descriptor: rd) else {
        report(label, false, "makeTexture nil for a shared \(w)x\(h) render target")
        return nil
    }
    return rt
}

// One pass over `rt`. `scissor` nil means the whole target.
func sharedTargetPass(_ rt: MTLTexture, _ pipe: MTLRenderPipelineState,
                      _ verts: MTLBuffer, _ colour: SIMD4<Float>,
                      load: Bool, clear: MTLClearColor, scissor: MTLScissorRect?) {
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = rt
    pass.colorAttachments[0].loadAction = load ? .load : .clear
    pass.colorAttachments[0].clearColor = clear
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(pipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    var c = colour
    enc.setFragmentBytes(&c, length: 16, index: 0)
    if let s = scissor { enc.setScissorRect(s) }
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

let magenta = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
let magentaTexel = pack(255, 0, 255, 255)
let greenTexel = pack(0, 255, 0, 255)
let blueTexel = pack(0, 0, 255, 255)

func sharedTargetCases(_ w: Int, _ h: Int) {
    let dims = "\(w)x\(h)"
    let names = (draw: "srt_draw_\(dims)",
                 load: "srt_load_preserves_\(dims)",
                 scissor: "srt_scissor_keeps_rest_\(dims)",
                 sample: "srt_sample_after_render_\(dims)")

    guard let pipe = solidPipeline else {
        for n in [names.draw, names.load, names.scissor, names.sample] {
            report(n, false, "solid pipeline unavailable")
        }
        return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // 1. Does a draw into guest-owned pages land at all.
    guard let rt = sharedRenderTarget(w, h, names.draw) else {
        for n in [names.load, names.scissor, names.sample] { skipDependent(n, names.draw) }
        return
    }
    sharedTargetPass(rt, pipe, verts, SIMD4<Float>(0, 1, 0, 1),
                     load: false, clear: magenta, scissor: nil)
    guard let drawn = readBack(readPipe, rt, w, h) else {
        refused(names.draw)
        for n in [names.load, names.scissor, names.sample] { skipDependent(n, names.draw) }
        return
    }
    var drawBad: [(Int, Int)] = []
    for y in 0..<h { for x in 0..<w where drawn[y * w + x] != greenTexel {
        drawBad.append((x, y)) } }
    report(names.draw, drawBad.isEmpty,
           drawBad.isEmpty ? "the whole guest-backed target is the drawn colour"
                           : "wrong=\(drawBad.count)/\(w * h) "
                             + "first=\(hex(drawn[drawBad[0].1 * w + drawBad[0].0])) "
                             + "want=\(hex(greenTexel))" + badMap(drawBad, w, h))

    // 2. A second pass with `.load` over the bottom half must leave the top
    //    half exactly as the first pass left it. A device that seeds `.load`
    //    from a copy taken before the first pass loses the top half here.
    guard let rt2 = sharedRenderTarget(w, h, names.load) else {
        for n in [names.scissor, names.sample] { skipDependent(n, names.load) }
        return
    }
    let half = h / 2
    sharedTargetPass(rt2, pipe, verts, SIMD4<Float>(0, 1, 0, 1), load: false,
                     clear: magenta, scissor: MTLScissorRect(x: 0, y: 0, width: w, height: half))
    sharedTargetPass(rt2, pipe, verts, SIMD4<Float>(0, 0, 1, 1), load: true,
                     clear: magenta,
                     scissor: MTLScissorRect(x: 0, y: half, width: w, height: h - half))
    if let got = readBack(readPipe, rt2, w, h) {
        var topBad: [(Int, Int)] = []
        var botBad: [(Int, Int)] = []
        for y in 0..<h {
            for x in 0..<w {
                let have = got[y * w + x]
                if y < half {
                    if have != greenTexel { topBad.append((x, y)) }
                } else if have != blueTexel { botBad.append((x, y)) }
            }
        }
        let ok = topBad.isEmpty && botBad.isEmpty
        report(names.load, ok,
               ok ? "the second pass added to what the first pass left"
                  : "first_pass_lost=\(topBad.count) second_pass_wrong=\(botBad.count)"
                    + badMap(topBad, w, h))
    } else {
        refused(names.load)
    }

    // 3. A scissored write into the middle of a loaded target. Only that
    //    rectangle may change; a stride the guest never agreed to shows up as
    //    the green landing on the wrong rows, which the untouched-count says.
    guard let rt3 = sharedRenderTarget(w, h, names.scissor) else {
        skipDependent(names.sample, names.scissor); return
    }
    sharedTargetPass(rt3, pipe, verts, SIMD4<Float>(1, 0, 1, 1),
                     load: false, clear: magenta, scissor: nil)
    let rx = w / 4, ry = h / 4
    let rw = max(1, w / 2), rh = max(1, h / 2)
    sharedTargetPass(rt3, pipe, verts, SIMD4<Float>(0, 1, 0, 1), load: true, clear: magenta,
                     scissor: MTLScissorRect(x: rx, y: ry, width: rw, height: rh))
    if let got = readBack(readPipe, rt3, w, h) {
        var inBad: [(Int, Int)] = []
        var outBad: [(Int, Int)] = []
        for y in 0..<h {
            for x in 0..<w {
                let inside = x >= rx && x < rx + rw && y >= ry && y < ry + rh
                let have = got[y * w + x]
                if inside {
                    if have != greenTexel { inBad.append((x, y)) }
                } else if have != magentaTexel { outBad.append((x, y)) }
            }
        }
        let ok = inBad.isEmpty && outBad.isEmpty
        report(names.scissor, ok,
               ok ? "the scissored write landed on exactly its own rows"
                  : "rect_wrong=\(inBad.count) outside_clobbered=\(outBad.count) "
                    + "rect=(\(rx),\(ry) \(rw)x\(rh))" + badMap(outBad, w, h))
    } else {
        refused(names.scissor)
    }

    // 4. The crossover: the same allocation, rendered into by one pass and
    //    sampled by the next. A device that keeps a target and a sampled view
    //    of one allocation as two images has to make the render visible to the
    //    read, and this is the case that says whether it did.
    guard let texPipe = texPipeline else {
        report(names.sample, false, "texture pipeline unavailable"); return
    }
    guard let src = sharedRenderTarget(w, h, names.sample) else { return }
    sharedTargetPass(src, pipe, verts, SIMD4<Float>(0, 1, 0, 1),
                     load: false, clear: magenta, scissor: nil)
    let dd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    dd.usage = [.renderTarget, .shaderRead]
    dd.storageMode = .private
    guard let dst = dev.makeTexture(descriptor: dd) else {
        report(names.sample, false, "makeTexture nil for the private destination"); return
    }
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = dst
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = magenta
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(texPipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    enc.setFragmentTexture(src, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let got = readBack(readPipe, dst, w, h) {
        var bad: [(Int, Int)] = []
        for y in 0..<h { for x in 0..<w where got[y * w + x] != greenTexel {
            bad.append((x, y)) } }
        report(names.sample, bad.isEmpty,
               bad.isEmpty ? "a later pass sampled what the earlier pass rendered"
                           : "wrong=\(bad.count)/\(w * h) "
                             + "first=\(hex(got[bad[0].1 * w + bad[0].0])) "
                             + "want=\(hex(greenTexel))" + badMap(bad, w, h))
    } else {
        refused(names.sample)
    }
}

sharedTargetCases(60, 32)
sharedTargetCases(256, 64)
sharedTargetCases(1000, 40)


// A guest-backed target whose content the CPU wrote before any GPU pass.
//
// This is the case the four above cannot reach. `srt_load_preserves` starts its
// first pass with `.clear`, so the target's prior content is this device's own
// and a rail that discards it on the way in still passes. A compositor's layer
// is the other order: CoreText rasterizes glyphs into the layer's bytes, and
// only then does the GPU composite over it with `.load`.
//
// A Vulkan image bound over memory that already holds data does not inherit
// that data -- the contents are undefined until something writes them -- so a
// device that aliases guest pages has to seed the first `.load` from those
// pages explicitly and may skip the seed on later ones. Both sides of that
// boundary are here: round 0 crosses the first-load seed, rounds 1 and 2 cross
// whatever the device does once it believes the image is authoritative.
//
// Everything CPU-written that a round did not draw over must survive every
// round. Losing it is a layer that renders its GPU geometry and drops the type
// the CPU put there, with no refusal anywhere.
func sharedTargetCpuSeedCase(_ w: Int, _ h: Int) {
    let label = "srt_cpu_seed_then_load_\(w)x\(h)"
    guard let pipe = solidPipeline else { report(label, false, "solid pipeline unavailable"); return }
    guard let rt = sharedRenderTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // 1. The CPU writes every texel, before the GPU has touched the target.
    let bpr = w * 4
    var cpu = [UInt8](repeating: 0, count: bpr * h)
    for y in 0..<h {
        for x in 0..<w {
            let o = y * bpr + x * 4
            cpu[o] = UInt8((x &* 7 &+ y &* 13 &+ 1) & 0xFF)
            cpu[o + 1] = UInt8((x &+ y &* 5) & 0xFF)
            cpu[o + 2] = 0x40
            cpu[o + 3] = 0xFF
        }
    }
    cpu.withUnsafeBytes { raw in
        rt.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                   withBytes: raw.baseAddress!, bytesPerRow: bpr)
    }

    // 2. Three `.load` passes, each over one horizontal band, leaving the last
    //    band CPU-only for the whole case.
    let band = max(1, h / 4)
    let drawn: [UInt32] = [greenTexel, blueTexel, pack(255, 0, 255, 255)]
    let colours: [SIMD4<Float>] = [SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1), SIMD4(1, 0, 1, 1)]
    for round in 0..<3 {
        sharedTargetPass(rt, pipe, verts, colours[round], load: true,
                         clear: magenta,
                         scissor: MTLScissorRect(x: 0, y: round * band,
                                                 width: w, height: band))
    }

    guard let got = readBack(readPipe, rt, w, h) else { refused(label); return }
    var gpuBad: [(Int, Int)] = []
    var cpuBad: [(Int, Int)] = []
    var firstCpu = ""
    for y in 0..<h {
        for x in 0..<w {
            let have = got[y * w + x]
            let round = y / band
            if round < 3 {
                if have != drawn[round] { gpuBad.append((x, y)) }
            } else {
                let o = y * bpr + x * 4
                let want = pack(cpu[o], cpu[o + 1], cpu[o + 2], cpu[o + 3])
                if have != want {
                    cpuBad.append((x, y))
                    if firstCpu.isEmpty {
                        firstCpu = "at=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))"
                    }
                }
            }
        }
    }
    let ok = gpuBad.isEmpty && cpuBad.isEmpty
    report(label, ok,
           ok ? "three loaded passes kept every texel the CPU wrote first"
              : "cpu_written_lost=\(cpuBad.count) gpu_bands_wrong=\(gpuBad.count) \(firstCpu) "
                + badMap(cpuBad, w, h))
}

sharedTargetCpuSeedCase(60, 32)
sharedTargetCpuSeedCase(256, 64)
sharedTargetCpuSeedCase(1000, 40)


// Type composited into a guest-backed layer.
//
// Everything in section I draws flat colour with blending off, and a driven
// Maps boot shows that is not where its type layer goes: it binds `R8Unorm`
// coverage atlases -- 128x128 and a scatter of 6x15, 10x1, 5x11 glyph
// bitmaps -- and blends them into a layer. Those atlases are sampled
// identically whether or not the device imports render targets, so whatever is
// lost is lost on the way *into* the layer, and this is the case shaped like
// that draw: a single-channel source, a premultiplied blend, and a
// guest-backed destination that accumulates across passes.
//
// The second pass is the half that a flat-colour case cannot stand in for. A
// text layer is built by blending over what is already there, so a rail that
// loses the destination on the way into a blended `.load` pass loses the type
// drawn before it while every solid draw in this file still passes.
func makeGlyphPipeline(_ fmt: MTLPixelFormat) -> MTLRenderPipelineState? {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = library.makeFunction(name: "quad_vs")
    d.fragmentFunction = library.makeFunction(name: "glyph_fs")
    let a = d.colorAttachments[0]!
    a.pixelFormat = fmt
    a.isBlendingEnabled = true
    a.rgbBlendOperation = .add
    a.alphaBlendOperation = .add
    a.sourceRGBBlendFactor = .one
    a.sourceAlphaBlendFactor = .one
    a.destinationRGBBlendFactor = .oneMinusSourceAlpha
    a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
    return try? dev.makeRenderPipelineState(descriptor: d)
}

let glyphPipeline = makeGlyphPipeline(.rgba8Unorm)

func sharedTargetGlyphCase(_ w: Int, _ h: Int) {
    let label = "srt_glyph_blend_\(w)x\(h)"
    guard let pipe = glyphPipeline else {
        report(label, false, "glyph pipeline unavailable"); return
    }
    // A single-channel coverage atlas in guest-visible storage, binary so the
    // expected blend has no rounding to argue about and a shear shows up as a
    // moved checker rather than as a slightly wrong colour.
    let ad = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r8Unorm, width: w, height: h, mipmapped: false)
    ad.usage = [.shaderRead]
    ad.storageMode = .shared
    guard let atlas = dev.makeTexture(descriptor: ad) else {
        report(label, false, "makeTexture nil for the coverage atlas"); return
    }
    var cov = [UInt8](repeating: 0, count: w * h)
    for y in 0..<h {
        for x in 0..<w { cov[y * w + x] = ((x / 3) + (y / 3)) % 2 == 0 ? 255 : 0 }
    }
    cov.withUnsafeBytes { raw in
        atlas.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                      withBytes: raw.baseAddress!, bytesPerRow: w)
    }

    guard let rt = sharedRenderTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func blend(_ colour: SIMD4<Float>, load: Bool, scissor: MTLScissorRect?) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = rt
        pass.colorAttachments[0].loadAction = load ? .load : .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        enc.setFragmentTexture(atlas, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        if let s = scissor { enc.setScissorRect(s) }
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // 1. Green type over an opaque black layer.
    blend(SIMD4<Float>(0, 1, 0, 1), load: false, scissor: nil)
    // 2. Blue type over the top half, loading what pass 1 left.
    let half = h / 2
    blend(SIMD4<Float>(0, 0, 1, 1), load: true,
          scissor: MTLScissorRect(x: 0, y: 0, width: w, height: half))

    guard let got = readBack(readPipe, rt, w, h) else { refused(label); return }
    let black = pack(0, 0, 0, 255)
    var bad: [(Int, Int)] = []
    var first = ""
    for y in 0..<h {
        for x in 0..<w {
            let on = cov[y * w + x] == 255
            let want: UInt32
            if !on {
                want = black
            } else if y < half {
                want = pack(0, 0, 255, 255)
            } else {
                want = pack(0, 255, 0, 255)
            }
            let have = got[y * w + x]
            if have != want {
                bad.append((x, y))
                if first.isEmpty { first = "at=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))" }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "coverage type blended into a guest-backed layer, and the second pass kept the first"
                       : "wrong=\(bad.count)/\(w * h) \(first)" + badMap(bad, w, h))
}

sharedTargetGlyphCase(60, 32)
sharedTargetGlyphCase(256, 64)
sharedTargetGlyphCase(1000, 40)


// ---------------------------------------------------------------------------
// J. Render targets backed by an IOSurface.
//
// Section I creates its guest-backed targets as plain `.shared` textures, and a
// driven Maps boot says that is not what its layers are: every `pass_target`
// this device logs carries a mapping id, which a plain shared texture never
// has. A layer that another process composites is an IOSurface, the texture is
// created over it with `makeTexture(descriptor:iosurface:plane:)`, and the
// device routes it through a different rail from the one section I exercises --
// its own resident registry, its own sample rung, its own serialized plane
// view.
//
// So section I's fifteen green cases say nothing about the rail Maps' type
// layer is actually composited on. These are the same four questions plus the
// blend, asked of the target the app really uses.
//
// The surface picks its own `bytesPerRow`, which is the point of asking at
// these widths: 60 and 1000 texels are 240 and 4000 bytes and IOSurface will
// pad both, so the texture's stride is one the test never chose and any rail
// that assumes a tight row has somewhere to go wrong.
func makeIOSurfaceTarget(_ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
    // 'BGRA' as an OSType, which is what IOSurface's pixel-format key takes.
    let bgra: UInt32 = 0x4247_5241
    let props: [IOSurfacePropertyKey: Any] = [
        .width: w,
        .height: h,
        .bytesPerElement: 4,
        .pixelFormat: bgra,
    ]
    guard let surface = IOSurface(properties: props) else {
        report(label, false, "IOSurface(properties:) nil for \(w)x\(h)")
        return nil
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .shared
    guard let tex = dev.makeTexture(descriptor: td, iosurface: surface, plane: 0) else {
        report(label, false, "makeTexture(iosurface:) nil for \(w)x\(h)")
        return nil
    }
    return tex
}

// BGRA in memory, so the packed comparison value has to swap to match
// `read_texels`, which reports the sampled RGBA.
func iosurfaceCases(_ w: Int, _ h: Int) {
    let dims = "\(w)x\(h)"
    let names = (draw: "iosrt_draw_\(dims)",
                 load: "iosrt_load_preserves_\(dims)",
                 scissor: "iosrt_scissor_keeps_rest_\(dims)",
                 glyph: "iosrt_glyph_blend_\(dims)")
    guard let solid = makeRenderPipeline("solid_fs", .bgra8Unorm) else {
        for n in [names.draw, names.load, names.scissor, names.glyph] {
            report(n, false, "solid pipeline unavailable for bgra8Unorm")
        }
        return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func pass(_ rt: MTLTexture, _ pipe: MTLRenderPipelineState,
              _ colour: SIMD4<Float>, load: Bool, scissor: MTLScissorRect?,
              atlas: MTLTexture?) {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = rt
        d.colorAttachments[0].loadAction = load ? .load : .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        if let atlas { enc.setFragmentTexture(atlas, index: 0) }
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        if let s = scissor { enc.setScissorRect(s) }
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // 1. A draw into an IOSurface-backed target.
    guard let rt = makeIOSurfaceTarget(w, h, names.draw) else {
        for n in [names.load, names.scissor, names.glyph] { skipDependent(n, names.draw) }
        return
    }
    pass(rt, solid, SIMD4<Float>(0, 1, 0, 1), load: false, scissor: nil, atlas: nil)
    guard let drawn = readBack(readPipe, rt, w, h) else {
        refused(names.draw)
        for n in [names.load, names.scissor, names.glyph] { skipDependent(n, names.draw) }
        return
    }
    var bad: [(Int, Int)] = []
    for y in 0..<h { for x in 0..<w where drawn[y * w + x] != greenTexel { bad.append((x, y)) } }
    report(names.draw, bad.isEmpty,
           bad.isEmpty ? "the whole IOSurface-backed target is the drawn colour"
                       : "wrong=\(bad.count)/\(w * h) "
                         + "first=\(hex(drawn[bad[0].1 * w + bad[0].0])) want=\(hex(greenTexel))"
                         + badMap(bad, w, h))

    // 2. A second pass over the bottom half must keep the first pass's top half.
    guard let rt2 = makeIOSurfaceTarget(w, h, names.load) else {
        for n in [names.scissor, names.glyph] { skipDependent(n, names.load) }
        return
    }
    let half = h / 2
    pass(rt2, solid, SIMD4<Float>(0, 1, 0, 1), load: false,
         scissor: MTLScissorRect(x: 0, y: 0, width: w, height: half), atlas: nil)
    pass(rt2, solid, SIMD4<Float>(0, 0, 1, 1), load: true,
         scissor: MTLScissorRect(x: 0, y: half, width: w, height: h - half), atlas: nil)
    if let got = readBack(readPipe, rt2, w, h) {
        var top: [(Int, Int)] = []
        var bot: [(Int, Int)] = []
        for y in 0..<h {
            for x in 0..<w {
                let have = got[y * w + x]
                if y < half { if have != greenTexel { top.append((x, y)) } }
                else if have != blueTexel { bot.append((x, y)) }
            }
        }
        let ok = top.isEmpty && bot.isEmpty
        report(names.load, ok,
               ok ? "the second pass added to what the first pass left"
                  : "first_pass_lost=\(top.count) second_pass_wrong=\(bot.count)"
                    + badMap(top, w, h))
    } else { refused(names.load) }

    // 3. A scissored write into the middle of a loaded IOSurface target.
    guard let rt3 = makeIOSurfaceTarget(w, h, names.scissor) else {
        skipDependent(names.glyph, names.scissor); return
    }
    pass(rt3, solid, SIMD4<Float>(1, 0, 1, 1), load: false, scissor: nil, atlas: nil)
    let rx = w / 4, ry = h / 4, rw = max(1, w / 2), rh = max(1, h / 2)
    pass(rt3, solid, SIMD4<Float>(0, 1, 0, 1), load: true,
         scissor: MTLScissorRect(x: rx, y: ry, width: rw, height: rh), atlas: nil)
    if let got = readBack(readPipe, rt3, w, h) {
        var inside: [(Int, Int)] = []
        var outside: [(Int, Int)] = []
        for y in 0..<h {
            for x in 0..<w {
                let within = x >= rx && x < rx + rw && y >= ry && y < ry + rh
                let have = got[y * w + x]
                if within { if have != greenTexel { inside.append((x, y)) } }
                else if have != magentaTexel { outside.append((x, y)) }
            }
        }
        let ok = inside.isEmpty && outside.isEmpty
        report(names.scissor, ok,
               ok ? "the scissored write landed on exactly its own rows"
                  : "rect_wrong=\(inside.count) outside_clobbered=\(outside.count) "
                    + "rect=(\(rx),\(ry) \(rw)x\(rh))" + badMap(outside, w, h))
    } else { refused(names.scissor) }

    // 4. Coverage type blended into the IOSurface layer, twice, the second
    //    loading the first. This is the draw Maps' type layer actually makes.
    guard let glyph = makeGlyphPipeline(.bgra8Unorm) else {
        report(names.glyph, false, "glyph pipeline unavailable for bgra8Unorm"); return
    }
    let ad = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r8Unorm, width: w, height: h, mipmapped: false)
    ad.usage = [.shaderRead]
    ad.storageMode = .shared
    guard let atlas = dev.makeTexture(descriptor: ad) else {
        report(names.glyph, false, "makeTexture nil for the coverage atlas"); return
    }
    var cov = [UInt8](repeating: 0, count: w * h)
    for y in 0..<h { for x in 0..<w { cov[y * w + x] = ((x / 3) + (y / 3)) % 2 == 0 ? 255 : 0 } }
    cov.withUnsafeBytes { raw in
        atlas.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                      withBytes: raw.baseAddress!, bytesPerRow: w)
    }
    guard let rt4 = makeIOSurfaceTarget(w, h, names.glyph) else { return }
    pass(rt4, glyph, SIMD4<Float>(0, 1, 0, 1), load: false, scissor: nil, atlas: atlas)
    pass(rt4, glyph, SIMD4<Float>(0, 0, 1, 1), load: true,
         scissor: MTLScissorRect(x: 0, y: 0, width: w, height: half), atlas: atlas)
    if let got = readBack(readPipe, rt4, w, h) {
        var wrong: [(Int, Int)] = []
        var first = ""
        for y in 0..<h {
            for x in 0..<w {
                let on = cov[y * w + x] == 255
                // A texel the type does not cover keeps this section's clear,
                // which is magenta. Expecting magenta rather than black is also
                // what makes the check say something: black is what a target
                // nobody wrote looks like, so a case that expected it would pass
                // on a layer that was never rendered into at all.
                let want = !on ? magentaTexel
                    : (y < half ? pack(0, 0, 255, 255) : pack(0, 255, 0, 255))
                let have = got[y * w + x]
                if have != want {
                    wrong.append((x, y))
                    if first.isEmpty { first = "at=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))" }
                }
            }
        }
        report(names.glyph, wrong.isEmpty,
               wrong.isEmpty
                 ? "coverage type blended into an IOSurface layer, and the second pass kept the first"
                 : "wrong=\(wrong.count)/\(w * h) \(first)" + badMap(wrong, w, h))
    } else { refused(names.glyph) }
}

iosurfaceCases(60, 32)
iosurfaceCases(256, 64)
iosurfaceCases(1000, 40)


// K. The CPU writes a layer the GPU already rendered into, and the GPU samples it.
//
// Section I and J both stop one step short of the shape a compositor actually
// runs. `cpu_write_after_render` writes the layer from the CPU and then reads
// it back with a compute kernel; `srt_sample_after_render` samples the layer but
// only ever sees texels the GPU itself drew. Neither one asks the question a
// text layer asks: after the CPU has written bytes into an allocation the GPU
// has already rendered into, does a *sampled bind* in a later render pass see
// them?
//
// The two reads are not interchangeable. A compute kernel reading a texture and
// a fragment shader sampling one are different binds, and a device that keeps a
// device-local image for an allocation it renders into may serve one from guest
// pages and the other from that image. A guest CPU store into its own RAM is
// invisible to the host -- no page fault, no command, nothing to observe -- so
// an image the device believes is authoritative stays stale with nothing
// anywhere reporting a loss.
//
// That is the whole failure mode this case exists for, and it is what a layer
// losing its type looks like: the GPU-drawn geometry in the layer is correct
// because the device drew it, and only the CPU-written half is missing.
func cpuWriteThenSampleCase(_ w: Int, _ h: Int, iosurface: Bool) {
    let rail = iosurface ? "iosrt" : "srt"
    let label = "\(rail)_cpu_write_then_sample_\(w)x\(h)"
    guard let pipe = iosurface ? makeRenderPipeline("solid_fs", .bgra8Unorm)
                               : solidPipeline else {
        report(label, false, "solid pipeline unavailable"); return
    }
    guard let texPipe = texPipeline else {
        report(label, false, "texture pipeline unavailable"); return
    }
    guard let layer = iosurface ? makeIOSurfaceTarget(w, h, label)
                                : sharedRenderTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // 1. The GPU renders the whole layer green. After this the device has an
    //    image for the allocation and every byte in it is the device's own.
    sharedTargetPass(layer, pipe, verts, SIMD4<Float>(0, 1, 0, 1),
                     load: false, clear: magenta, scissor: nil)

    // 2. The CPU writes the bottom half opaque red, the way CoreText rasterizes
    //    glyphs into a layer's bytes. `.shared` is BGRA in memory on the
    //    IOSurface rail and RGBA on the other, so the channel order follows the
    //    format rather than the case.
    let half = max(1, h / 2)
    let rows = h - half
    let bpr = w * 4
    var cpu = [UInt8](repeating: 0, count: bpr * rows)
    for i in 0..<(w * rows) {
        let o = i * 4
        if iosurface { cpu[o] = 0; cpu[o + 1] = 0; cpu[o + 2] = 255 }
        else { cpu[o] = 255; cpu[o + 1] = 0; cpu[o + 2] = 0 }
        cpu[o + 3] = 255
    }
    cpu.withUnsafeBytes { raw in
        layer.replace(region: MTLRegionMake2D(0, half, w, rows), mipmapLevel: 0,
                      withBytes: raw.baseAddress!, bytesPerRow: bpr)
    }

    // 3. A later render pass samples the layer into a private destination --
    //    the compositor's read, not a compute read-back of the layer itself.
    let dd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    dd.usage = [.renderTarget, .shaderRead]
    dd.storageMode = .private
    guard let dst = dev.makeTexture(descriptor: dd) else {
        report(label, false, "makeTexture nil for the private destination"); return
    }
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = dst
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = magenta
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(texPipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    enc.setFragmentTexture(layer, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    guard let got = readBack(readPipe, dst, w, h) else { refused(label); return }
    let redTexel = pack(255, 0, 0, 255)
    var gpuBad: [(Int, Int)] = []
    var cpuBad: [(Int, Int)] = []
    var firstCpu = ""
    for y in 0..<h {
        for x in 0..<w {
            let have = got[y * w + x]
            if y < half {
                if have != greenTexel { gpuBad.append((x, y)) }
            } else if have != redTexel {
                cpuBad.append((x, y))
                if firstCpu.isEmpty {
                    firstCpu = "at=(\(x),\(y)) want=\(hex(redTexel)) got=\(hex(have))"
                }
            }
        }
    }
    let ok = gpuBad.isEmpty && cpuBad.isEmpty
    report(label, ok,
           ok ? "a sampled bind saw the texels the CPU wrote after the GPU rendered"
              : "cpu_written_unseen=\(cpuBad.count)/\(w * rows) "
                + "gpu_drawn_wrong=\(gpuBad.count)/\(w * half) \(firstCpu)"
                + badMap(cpuBad, w, h))
}

for (cw, ch) in [(60, 32), (256, 64), (1000, 40)] {
    cpuWriteThenSampleCase(cw, ch, iosurface: false)
    cpuWriteThenSampleCase(cw, ch, iosurface: true)
}


// L. A whole-surface blit of a target the GPU has only just rendered into.
//
// The guest never reads the layer here — it hands both endpoints to a blit and
// asks the device to move the pixels. Metal orders that copy against the render
// before it: two command buffers on one queue execute in the order they were
// committed, so the copy sees everything the render wrote.
//
// A device that decomposes the pair does not get that ordering for free. If the
// render is submitted to the GPU and the copy is then serviced by reading the
// source's bytes on the CPU, the two are racing, and the reader wins whenever
// the GPU has not finished — which is most of the time, because submission is
// asynchronous and the copy is decoded immediately behind it.
//
// # The shape is measured, not chosen
//
// Which copies the guest driver emits as a whole-surface texture-to-texture
// copy, rather than staging through a buffer, is the driver's decision and not
// something this file can assume. Earlier attempts at this case used a
// `.bgra8Unorm` pair at 512x512 and never produced one: the driver staged every
// one of them, so the case exercised a different rail and passed for the wrong
// reason.
//
// The shape below is what a driven compositor actually emits — a linear source,
// an IOSurface-backed destination, `BGRA8Unorm_sRGB`, one level and one slice,
// at window and screen size. That is the pair a compositor produces when it
// draws a layer and then copies it into the surface the window server owns.
//
// # Why two passes
//
// The first pass lands red and is waited on, so the source's bytes are known
// and are *not* the answer. The second lands green and is not waited on before
// the copy. A correct device puts green in the destination; one that reads the
// source without ordering puts **red** there — the previous frame, whole and
// undamaged, which is why this class reads as a layer showing stale content
// rather than as corruption. `stale_previous_frame` in the failure counts
// exactly those texels, so the report distinguishes this defect from a copy
// that simply moved nothing.
//
// Full-intensity red and green round-trip an 8-bit sRGB encode exactly (0 and 1
// are both fixed points), so the expectation is unaffected by the transfer
// function on the attachment.
func makeSrgbIOSurfaceTarget(_ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
    let bgra: UInt32 = 0x4247_5241   // 'BGRA' as an OSType
    let props: [IOSurfacePropertyKey: Any] = [
        .width: w, .height: h, .bytesPerElement: 4, .pixelFormat: bgra,
    ]
    guard let surface = IOSurface(properties: props) else {
        report(label, false, "IOSurface(properties:) nil for \(w)x\(h)"); return nil
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .shared
    guard let tex = dev.makeTexture(descriptor: td, iosurface: surface, plane: 0) else {
        report(label, false, "makeTexture(iosurface:) nil for \(w)x\(h)"); return nil
    }
    return tex
}

func blitAfterRenderCase(_ w: Int, _ h: Int) {
    let label = "srt_blit_after_render_\(w)x\(h)"
    guard let pipe = makeRenderPipeline("heavy_fs", .bgra8Unorm_srgb) else {
        report(label, false, "heavy pipeline unavailable for bgra8Unorm_srgb"); return
    }
    let sd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
    sd.usage = [.renderTarget, .shaderRead]
    sd.storageMode = .shared
    guard let source = dev.makeTexture(descriptor: sd) else {
        report(label, false, "makeTexture nil for the shared \(w)x\(h) source"); return
    }
    guard let dst = makeSrgbIOSurfaceTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func encodeFill(_ cb: MTLCommandBuffer, _ colour: SIMD4<Float>, draws: Int) {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = source
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let enc = cb.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        // Repeated whole-target draws, so the GPU is still working when the copy
        // behind them is decoded. One draw would leave the race to scheduling
        // luck and make the case flaky in the direction that reads as a pass.
        for _ in 0..<draws {
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        enc.endEncoding()
    }

    // 1. Red, landed. After this the source's bytes are red and nothing else.
    let first = queue.makeCommandBuffer()!
    encodeFill(first, SIMD4<Float>(1, 0, 0, 1), draws: 1)
    first.commit()
    first.waitUntilCompleted()

    // 2. Green, committed and not waited on, then the copy in its own command
    //    buffer. Queue order is what orders them; nothing else has to.
    let second = queue.makeCommandBuffer()!
    encodeFill(second, SIMD4<Float>(0, 1, 0, 1), draws: 256)
    second.commit()

    let copyCb = queue.makeCommandBuffer()!
    let blit = copyCb.makeBlitCommandEncoder()!
    blit.copy(from: source, sourceSlice: 0, sourceLevel: 0,
              to: dst, destinationSlice: 0, destinationLevel: 0,
              sliceCount: 1, levelCount: 1)
    blit.endEncoding()
    copyCb.commit()
    copyCb.waitUntilCompleted()

    guard let got = readBack(readPipe, dst, w, h) else { refused(label); return }
    let green = pack(0, 255, 0, 255)
    let red = pack(255, 0, 0, 255)
    var wrong: [(Int, Int)] = []
    var stale = 0
    for y in 0..<h {
        for x in 0..<w where got[y * w + x] != green {
            wrong.append((x, y))
            if got[y * w + x] == red { stale += 1 }
        }
    }
    let firstBad = wrong.first.map {
        "at=(\($0.0),\($0.1)) got=\(hex(got[$0.1 * w + $0.0]))"
    } ?? ""
    report(label, wrong.isEmpty,
           wrong.isEmpty
             ? "the copy moved what the render ahead of it had written"
             : "wrong=\(wrong.count)/\(w * h) stale_previous_frame=\(stale) "
               + "want=\(hex(green)) \(firstBad)" + badMap(wrong, w, h))
}

blitAfterRenderCase(1024, 768)
blitAfterRenderCase(1920, 1080)

// The same copy, run the way a compositor runs it: many frames in flight.
//
// The single-shot case above reaches the rail and does not fail, because one
// render is not enough to keep the GPU busy past the point where the copy
// behind it is serviced. A compositor never has one frame outstanding — it
// renders a layer, copies it out, and starts the next without waiting for
// either, so a copy is decoded while earlier frames are still executing and the
// queue is never drained.
//
// Each frame gets its own colour and its own destination, so a stale read is not
// merely "wrong" — it is identifiably *an earlier frame*, which is what the
// report names. Nothing is waited on until every frame has been committed.
func blitPipelinedCase(_ w: Int, _ h: Int, frames: Int) {
    let label = "srt_blit_pipelined_\(w)x\(h)_x\(frames)"
    guard let pipe = makeRenderPipeline("heavy_fs", .bgra8Unorm_srgb) else {
        report(label, false, "heavy pipeline unavailable for bgra8Unorm_srgb"); return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // One colour per frame, all full-intensity so the sRGB encode round-trips
    // exactly, and all distinguishable from one another.
    let palette: [(SIMD4<Float>, UInt32)] = [
        (SIMD4(1, 0, 0, 1), pack(255, 0, 0, 255)),
        (SIMD4(0, 1, 0, 1), pack(0, 255, 0, 255)),
        (SIMD4(0, 0, 1, 1), pack(0, 0, 255, 255)),
        (SIMD4(1, 1, 0, 1), pack(255, 255, 0, 255)),
        (SIMD4(1, 0, 1, 1), pack(255, 0, 255, 255)),
        (SIMD4(0, 1, 1, 1), pack(0, 255, 255, 255)),
        (SIMD4(1, 1, 1, 1), pack(255, 255, 255, 255)),
        (SIMD4(0, 0, 0, 1), pack(0, 0, 0, 255)),
    ]

    var sources: [MTLTexture] = []
    var dests: [MTLTexture] = []
    for i in 0..<frames {
        let sd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
        sd.usage = [.renderTarget, .shaderRead]
        sd.storageMode = .shared
        guard let src = dev.makeTexture(descriptor: sd) else {
            report(label, false, "makeTexture nil for frame \(i)'s source"); return
        }
        guard let dst = makeSrgbIOSurfaceTarget(w, h, label) else { return }
        sources.append(src)
        dests.append(dst)
    }

    var last: MTLCommandBuffer?
    for i in 0..<frames {
        let (colour, _) = palette[i % palette.count]
        let render = queue.makeCommandBuffer()!
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = sources[i]
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let enc = render.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        for _ in 0..<96 {
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        enc.endEncoding()
        render.commit()

        let copyCb = queue.makeCommandBuffer()!
        let blit = copyCb.makeBlitCommandEncoder()!
        blit.copy(from: sources[i], sourceSlice: 0, sourceLevel: 0,
                  to: dests[i], destinationSlice: 0, destinationLevel: 0,
                  sliceCount: 1, levelCount: 1)
        blit.endEncoding()
        copyCb.commit()
        last = copyCb
    }
    last?.waitUntilCompleted()

    var badFrames: [Int] = []
    var detail = ""
    for i in 0..<frames {
        guard let got = readBack(readPipe, dests[i], w, h) else { refused(label); return }
        let want = palette[i % palette.count].1
        var wrong = 0
        var sawOtherFrame = -1
        for t in 0..<(w * h) where got[t] != want {
            wrong += 1
            if sawOtherFrame < 0 {
                for (j, p) in palette.enumerated() where p.1 == got[t] { sawOtherFrame = j }
            }
        }
        if wrong > 0 {
            badFrames.append(i)
            if detail.isEmpty {
                detail = "frame=\(i) wrong=\(wrong)/\(w * h) want=\(hex(want)) "
                    + (sawOtherFrame >= 0
                        ? "got=another frame's colour (palette \(sawOtherFrame))"
                        : "got=\(hex(got[0]))")
            }
        }
    }
    report(label, badFrames.isEmpty,
           badFrames.isEmpty
             ? "every frame's copy moved that frame's own pixels"
             : "stale_frames=\(badFrames.count)/\(frames) \(detail)")
}

blitPipelinedCase(1024, 768, frames: 8)

// A buffer-backed source, which is a linear guest allocation by construction.
//
// A texture made from an `MTLBuffer` names its own base and row stride, so
// unlike a plain `.shared` texture — whose layout the driver picks — it is
// unambiguously the linear form. Both are worth having: the pair above is what
// a compositor emits, and this is the shape that leaves the guest no room to
// choose something else.
//
// # This is a second defect, and it is not the unordered read
//
// The case fails with every texel zero and `stale_previous_frame=0`, which looks
// at first like the unordered read above — zero is also what pre-Store bytes
// look like on a freshly allocated source. It is not. It fails identically with
// the ordering in place, and the first pass here is waited on, so a racing copy
// would have found red rather than nothing.
//
// The device names it on the fail channel:
//
//     rt_resolve reason=rt_wrong_type object_type=texture_view
//
// A texture made from an `MTLBuffer` arrives as a texture *view*, and the
// render-target resolver does not accept that object type as a colour
// attachment. Every draw into one is dropped, so the guest reads back the
// allocation untouched. Metal renders into it happily, which is why this case
// passes natively and fails here.
//
// The refusal is correct behaviour for an unimplemented case — it costs the
// guest one command and says so by name, rather than guessing. The case stays
// red until the resolver accepts the type.
func makeLinearTarget(_ w: Int, _ h: Int, _ label: String,
                      _ format: MTLPixelFormat) -> MTLTexture? {
    let align = max(1, dev.minimumLinearTextureAlignment(for: format))
    let bpr = ((w * 4) + align - 1) / align * align
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(label, false, "makeBuffer nil for a linear \(w)x\(h) target"); return nil
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: format, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .shared
    guard let tex = buf.makeTexture(descriptor: td, offset: 0, bytesPerRow: bpr) else {
        report(label, false, "makeTexture nil for a buffer-backed \(w)x\(h) render target")
        return nil
    }
    return tex
}

func blitBufferBackedCase(_ w: Int, _ h: Int) {
    let label = "srt_blit_buffer_backed_\(w)x\(h)"
    guard let pipe = makeRenderPipeline("solid_fs", .bgra8Unorm) else {
        report(label, false, "solid pipeline unavailable for bgra8Unorm"); return
    }
    guard let source = makeLinearTarget(w, h, label, .bgra8Unorm) else { return }
    guard let dst = makeIOSurfaceTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func encodeFill(_ cb: MTLCommandBuffer, _ colour: SIMD4<Float>, draws: Int) {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = source
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let enc = cb.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        for _ in 0..<draws {
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        enc.endEncoding()
    }

    let first = queue.makeCommandBuffer()!
    encodeFill(first, SIMD4<Float>(1, 0, 0, 1), draws: 1)
    first.commit()
    first.waitUntilCompleted()

    let second = queue.makeCommandBuffer()!
    encodeFill(second, SIMD4<Float>(0, 1, 0, 1), draws: 64)
    second.commit()

    let copyCb = queue.makeCommandBuffer()!
    let blit = copyCb.makeBlitCommandEncoder()!
    blit.copy(from: source, sourceSlice: 0, sourceLevel: 0,
              to: dst, destinationSlice: 0, destinationLevel: 0,
              sliceCount: 1, levelCount: 1)
    blit.endEncoding()
    copyCb.commit()
    copyCb.waitUntilCompleted()

    guard let got = readBack(readPipe, dst, w, h) else { refused(label); return }
    let green = pack(0, 255, 0, 255)
    let red = pack(255, 0, 0, 255)
    var wrong: [(Int, Int)] = []
    var stale = 0
    var zero = 0
    for y in 0..<h {
        for x in 0..<w where got[y * w + x] != green {
            wrong.append((x, y))
            if got[y * w + x] == red { stale += 1 }
            if got[y * w + x] == 0 { zero += 1 }
        }
    }
    let firstBad = wrong.first.map {
        "at=(\($0.0),\($0.1)) got=\(hex(got[$0.1 * w + $0.0]))"
    } ?? ""
    report(label, wrong.isEmpty,
           wrong.isEmpty
             ? "the copy out of a buffer-backed source moved what the render had written"
             : "wrong=\(wrong.count)/\(w * h) stale_previous_frame=\(stale) "
               + "never_written=\(zero) want=\(hex(green)) \(firstBad)" + badMap(wrong, w, h))
}

blitBufferBackedCase(512, 512)

print("SUMMARY cases=\(ran) failures=\(failures) skipped=\(skipped)")
print("DEVICE name=\(dev.name) unified=\(dev.hasUnifiedMemory)")
exit(failures == 0 ? 0 : 1)
