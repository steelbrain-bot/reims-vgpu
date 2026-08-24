import Metal
import Foundation

// A texture view owns the texture generation it was constructed from. Releasing
// the application's base reference and creating another texture must not
// redirect the already-created view to the replacement object.
func textureViewLifetimeCase() {
    let label = "texture_view_retains_base_generation"
    let w = 8, h = 8
    let desc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    desc.storageMode = .shared
    desc.usage = [.shaderRead, .pixelFormatView]

    var base: MTLTexture? = dev.makeTexture(descriptor: desc)
    guard base != nil else {
        report(label, false, "base allocation failed"); return
    }
    let originalPixel: [UInt8] = [17, 83, 149, 255]
    let originalBytes = Array(repeating: originalPixel, count: w * h).flatMap { $0 }
    base!.replace(
        region: MTLRegionMake2D(0, 0, w, h),
        mipmapLevel: 0,
        withBytes: originalBytes,
        bytesPerRow: w * 4)
    guard let view = base!.makeTextureView(pixelFormat: .rgba8Unorm) else {
        report(label, false, "view construction failed"); return
    }

    // Drop the only application-owned base reference. `view` is still live and
    // therefore still names the original texture generation.
    base = nil
    guard let replacement = dev.makeTexture(descriptor: desc) else {
        report(label, false, "replacement allocation failed"); return
    }
    let replacementPixel: [UInt8] = [211, 37, 19, 255]
    let replacementBytes = Array(repeating: replacementPixel, count: w * h).flatMap { $0 }
    replacement.replace(
        region: MTLRegionMake2D(0, 0, w, h),
        mipmapLevel: 0,
        withBytes: replacementBytes,
        bytesPerRow: w * 4)

    guard let got = readBack(readPipe, view, w, h) else {
        refused(label); return
    }
    let want = pack(originalPixel[0], originalPixel[1], originalPixel[2], originalPixel[3])
    let replacementValue = pack(
        replacementPixel[0], replacementPixel[1], replacementPixel[2], replacementPixel[3])
    let wrong = got.filter { $0 != want }
    let redirected = got.filter { $0 == replacementValue }.count
    report(
        label,
        wrong.isEmpty,
        wrong.isEmpty
            ? "the live view kept all \(w * h) texels from its construction-time base"
            : "\(wrong.count)/\(w * h) wrong redirected_to_replacement=\(redirected) first=\(hex(got[0])) want=\(hex(want))")
}
