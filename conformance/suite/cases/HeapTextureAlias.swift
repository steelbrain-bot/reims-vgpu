import Metal
import Foundation

private func clearHeapTexture(_ texture: MTLTexture, _ color: MTLClearColor) -> Bool {
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = texture
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = color
    pass.colorAttachments[0].storeAction = .store

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
        return false
    }
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.status == .completed
}

// A one-slot automatic heap has exactly one legal reuse transition: after the
// live texture occupying it becomes aliasable, a second compatible texture may
// take the same offset. The final readback covers both halves of the contract:
// the allocator reused the released range, and commands address the new
// resource rather than stale storage owned by the old one.
func heapTextureAliasCase() {
    let label = "heap_texture_alias_lifecycle"
    let width = 257
    let height = 193
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: width,
        height: height,
        mipmapped: false)
    descriptor.storageMode = .private
    descriptor.usage = [.renderTarget, .shaderRead]

    let requirement = dev.heapTextureSizeAndAlign(descriptor: descriptor)
    guard requirement.size > 0, requirement.align > 0 else {
        skip(label, "the device reported no heap storage requirement for the texture")
        return
    }
    guard requirement.align.nonzeroBitCount == 1 else {
        report(label, false, "heap alignment is not a power of two: \(requirement.align)")
        return
    }

    let heapDescriptor = MTLHeapDescriptor()
    heapDescriptor.type = .automatic
    heapDescriptor.storageMode = .private
    heapDescriptor.hazardTrackingMode = .untracked
    heapDescriptor.size = requirement.size
    guard let heap = dev.makeHeap(descriptor: heapDescriptor) else {
        report(label, false,
               "nonzero requirement size=\(requirement.size) align=\(requirement.align), but makeHeap returned nil")
        return
    }
    guard let first = heap.makeTexture(descriptor: descriptor) else {
        report(label, false, "the exact-size heap would not allocate its first texture")
        return
    }
    let firstOffset = first.heapOffset
    guard heap.makeTexture(descriptor: descriptor) == nil else {
        report(label, false, "a one-slot heap admitted two simultaneously live textures")
        return
    }
    guard clearHeapTexture(first, MTLClearColor(red: 1, green: 0, blue: 0, alpha: 1)) else {
        report(label, false, "commands using the first heap texture did not complete")
        return
    }

    first.makeAliasable()
    guard first.isAliasable() else {
        report(label, false, "makeAliasable did not change the first texture's lifecycle state")
        return
    }
    guard let second = heap.makeTexture(descriptor: descriptor) else {
        report(label, false, "the released heap range was not reusable")
        return
    }
    guard second.heapOffset == firstOffset else {
        report(label, false,
               "the replacement moved from offset=\(firstOffset) to offset=\(second.heapOffset)")
        return
    }
    guard clearHeapTexture(second, MTLClearColor(red: 0, green: 1, blue: 0, alpha: 1)) else {
        report(label, false, "commands using the replacement heap texture did not complete")
        return
    }
    guard let got = readBack(readPipe, second, width, height) else {
        refused(label)
        return
    }

    let want = pack(0, 255, 0, 255)
    let bad = got.indices.filter { got[$0] != want }
    report(label, bad.isEmpty,
           bad.isEmpty
             ? "one live range was released, reused at offset=\(second.heapOffset), and addressed as the replacement"
             : "wrong=\(bad.count)/\(got.count) first=\(hex(got[bad[0]])) want=\(hex(want))")
}
