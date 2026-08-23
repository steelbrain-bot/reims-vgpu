import Metal
import Foundation

private func depthPipeline(_ depthFormat: MTLPixelFormat,
                           stencil stencilFormat: MTLPixelFormat = .invalid) -> MTLRenderPipelineState? {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "quad_vs")
    descriptor.fragmentFunction = library.makeFunction(name: "solid_fs")
    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    descriptor.depthAttachmentPixelFormat = depthFormat
    descriptor.stencilAttachmentPixelFormat = stencilFormat
    return try? dev.makeRenderPipelineState(descriptor: descriptor)
}

private func stencilTestState(_ compare: MTLCompareFunction) -> MTLDepthStencilState? {
    let face = MTLStencilDescriptor()
    face.stencilCompareFunction = compare
    face.stencilFailureOperation = .keep
    face.depthFailureOperation = .keep
    face.depthStencilPassOperation = .keep
    face.readMask = UInt32.max
    face.writeMask = 0
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .always
    descriptor.isDepthWriteEnabled = false
    descriptor.frontFaceStencil = face
    descriptor.backFaceStencil = face
    return dev.makeDepthStencilState(descriptor: descriptor)
}

private func depthTestState(_ compare: MTLCompareFunction) -> MTLDepthStencilState? {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = compare
    descriptor.isDepthWriteEnabled = false
    return dev.makeDepthStencilState(descriptor: descriptor)
}

/// Pass attachment operations and per-draw tests are independent Metal state.
///
/// The first case clears/stores depth while no depth-stencil state is bound,
/// then consumes that clear in a later pass. The second binds a rejecting depth
/// state to a pass with no depth attachment; Metal disables the absent test, so
/// the colour draw still lands.
func depthStencilIndependenceCases() {
    let width = 8, height = 8
    let vertices = dev.makeBuffer(bytes: quadVerts,
                                  length: quadVerts.count * MemoryLayout<Float>.size,
                                  options: .storageModeShared)!
    let colourDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
    colourDescriptor.usage = [.renderTarget, .shaderRead]
    colourDescriptor.storageMode = .private

    if let pipeline = depthPipeline(.depth32Float),
       let less = depthTestState(.less),
       let colour = dev.makeTexture(descriptor: colourDescriptor) {
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
        depthDescriptor.usage = .renderTarget
        depthDescriptor.storageMode = .private

        if let depth = dev.makeTexture(descriptor: depthDescriptor) {
            var firstColour: [Float] = [1, 0, 0, 1]
            let first = MTLRenderPassDescriptor()
            first.colorAttachments[0].texture = colour
            first.colorAttachments[0].loadAction = .clear
            first.colorAttachments[0].storeAction = .dontCare
            first.depthAttachment.texture = depth
            first.depthAttachment.loadAction = .clear
            first.depthAttachment.storeAction = .store
            first.depthAttachment.clearDepth = 0.0

            let commandBuffer = queue.makeCommandBuffer()!
            let firstEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: first)!
            firstEncoder.setRenderPipelineState(pipeline)
            firstEncoder.setVertexBuffer(vertices, offset: 0, index: 0)
            firstEncoder.setFragmentBytes(&firstColour, length: 16, index: 0)
            // Deliberately leave the depth-stencil state nil. The pass-owned
            // clear/store must still execute.
            firstEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            firstEncoder.endEncoding()

            var secondColour: [Float] = [0, 1, 0, 1]
            let second = MTLRenderPassDescriptor()
            second.colorAttachments[0].texture = colour
            second.colorAttachments[0].loadAction = .clear
            second.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 1, alpha: 1)
            second.colorAttachments[0].storeAction = .store
            second.depthAttachment.texture = depth
            second.depthAttachment.loadAction = .load
            second.depthAttachment.storeAction = .dontCare

            let secondEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: second)!
            secondEncoder.setRenderPipelineState(pipeline)
            secondEncoder.setDepthStencilState(less)
            secondEncoder.setVertexBuffer(vertices, offset: 0, index: 0)
            secondEncoder.setFragmentBytes(&secondColour, length: 16, index: 0)
            // quad_vs emits z=0. Less-than the stored clear value 0 is false,
            // so the blue pass clear must survive.
            secondEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            secondEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if let pixels = readBack(readPipe, colour, width, height) {
                let expected = pack(0, 0, 255, 255)
                let ok = pixels.allSatisfy { $0 == expected }
                report("depth_attachment_clear_without_test_state", ok,
                       ok ? "stored depth clear rejected the later draw"
                          : "want=\(hex(expected)) got=\(hex(pixels[0]))")
            } else {
                refused("depth_attachment_clear_without_test_state")
            }
        } else {
            report("depth_attachment_clear_without_test_state", false,
                   "depth32Float render-target allocation failed")
        }
    } else {
        report("depth_attachment_clear_without_test_state", false,
               "depth pipeline, state, or colour target creation failed")
    }

    if let pipeline = depthPipeline(.invalid) {
        // Native Metal applies a bound depth comparison even without a pass
        // depth attachment. For quad_vs' z=0 the implicit comparison value is
        // 1: Less and Always pass; Equal, Greater and Never reject.
        let comparisons: [(String, MTLCompareFunction, Bool)] = [
            ("less", .less, true),
            ("equal", .equal, false),
            ("greater", .greater, false),
            ("never", .never, false),
            ("always", .always, true),
        ]
        for (tag, compare, shouldDraw) in comparisons {
            let label = "depth_test_\(tag)_without_attachment"
            guard let state = depthTestState(compare),
                  let colour = dev.makeTexture(descriptor: colourDescriptor) else {
                report(label, false, "depth state or colour target creation failed")
                continue
            }
            var green: [Float] = [0, 1, 0, 1]
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = colour
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 1, alpha: 1)
            pass.colorAttachments[0].storeAction = .store

            let commandBuffer = queue.makeCommandBuffer()!
            let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)!
            encoder.setRenderPipelineState(pipeline)
            encoder.setDepthStencilState(state)
            encoder.setVertexBuffer(vertices, offset: 0, index: 0)
            encoder.setFragmentBytes(&green, length: 16, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if let pixels = readBack(readPipe, colour, width, height) {
                let expected = shouldDraw ? pack(0, 255, 0, 255) : pack(0, 0, 255, 255)
                let ok = pixels.allSatisfy { $0 == expected }
                report(label, ok,
                       ok ? "z=0 compared against the absent attachment value"
                          : "want=\(hex(expected)) got=\(hex(pixels[0]))")
            } else {
                refused(label)
            }
        }
    } else {
        for tag in ["less", "equal", "greater", "never", "always"] {
            report("depth_test_\(tag)_without_attachment", false,
                   "colour-only pipeline creation failed")
        }
    }
}

func stencilIndependenceCases() {
    let width = 8, height = 8
    let vertices = dev.makeBuffer(bytes: quadVerts,
                                  length: quadVerts.count * MemoryLayout<Float>.size,
                                  options: .storageModeShared)!
    let colourDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
    colourDescriptor.usage = [.renderTarget, .shaderRead]
    colourDescriptor.storageMode = .private

    let combinedFormat = MTLPixelFormat.depth32Float_stencil8
    if let pipeline = depthPipeline(combinedFormat, stencil: combinedFormat),
       let equal = stencilTestState(.equal),
       let colour = dev.makeTexture(descriptor: colourDescriptor) {
        let attachmentDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: combinedFormat, width: width, height: height, mipmapped: false)
        attachmentDescriptor.usage = .renderTarget
        attachmentDescriptor.storageMode = .private
        if let attachment = dev.makeTexture(descriptor: attachmentDescriptor) {
            var red: [Float] = [1, 0, 0, 1]
            let first = MTLRenderPassDescriptor()
            first.colorAttachments[0].texture = colour
            first.colorAttachments[0].loadAction = .clear
            first.colorAttachments[0].storeAction = .dontCare
            first.depthAttachment.texture = attachment
            first.depthAttachment.loadAction = .clear
            first.depthAttachment.storeAction = .store
            first.depthAttachment.clearDepth = 1.0
            first.stencilAttachment.texture = attachment
            first.stencilAttachment.loadAction = .clear
            first.stencilAttachment.storeAction = .store
            first.stencilAttachment.clearStencil = 7

            let commandBuffer = queue.makeCommandBuffer()!
            let firstEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: first)!
            firstEncoder.setRenderPipelineState(pipeline)
            firstEncoder.setVertexBuffer(vertices, offset: 0, index: 0)
            firstEncoder.setFragmentBytes(&red, length: 16, index: 0)
            firstEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            firstEncoder.endEncoding()

            var green: [Float] = [0, 1, 0, 1]
            let second = MTLRenderPassDescriptor()
            second.colorAttachments[0].texture = colour
            second.colorAttachments[0].loadAction = .clear
            second.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 1, alpha: 1)
            second.colorAttachments[0].storeAction = .store
            second.depthAttachment.texture = attachment
            second.depthAttachment.loadAction = .load
            second.depthAttachment.storeAction = .dontCare
            second.stencilAttachment.texture = attachment
            second.stencilAttachment.loadAction = .load
            second.stencilAttachment.storeAction = .dontCare

            let secondEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: second)!
            secondEncoder.setRenderPipelineState(pipeline)
            secondEncoder.setDepthStencilState(equal)
            secondEncoder.setStencilReferenceValue(7)
            secondEncoder.setVertexBuffer(vertices, offset: 0, index: 0)
            secondEncoder.setFragmentBytes(&green, length: 16, index: 0)
            secondEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            secondEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if let pixels = readBack(readPipe, colour, width, height) {
                let expected = pack(0, 255, 0, 255)
                let ok = pixels.allSatisfy { $0 == expected }
                report("stencil_attachment_clear_without_test_state", ok,
                       ok ? "stored stencil clear passed the later equal test"
                          : "want=\(hex(expected)) got=\(hex(pixels[0]))")
            } else {
                refused("stencil_attachment_clear_without_test_state")
            }
        } else {
            report("stencil_attachment_clear_without_test_state", false,
                   "combined depth-stencil allocation failed")
        }
    } else {
        report("stencil_attachment_clear_without_test_state", false,
               "combined pipeline, state, or colour target creation failed")
    }

    let variants: [(String, MTLCompareFunction, Bool)] = [
        ("equal", .equal, true),
        ("not_equal", .notEqual, false),
    ]
    for hasDepth in [false, true] {
        let depthFormat: MTLPixelFormat = hasDepth ? .depth32Float : .invalid
        guard let pipeline = depthPipeline(depthFormat) else {
            for (tag, _, _) in variants {
                report("stencil_test_\(tag)_without_attachment_depth_\(hasDepth)", false,
                       "pipeline creation failed")
            }
            continue
        }
        let depth: MTLTexture? = {
            guard hasDepth else { return nil }
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .depth32Float, width: width, height: height, mipmapped: false)
            descriptor.usage = .renderTarget
            descriptor.storageMode = .private
            return dev.makeTexture(descriptor: descriptor)
        }()
        for (tag, compare, shouldDraw) in variants {
            let label = "stencil_test_\(tag)_without_attachment_depth_\(hasDepth)"
            guard let state = stencilTestState(compare),
                  let colour = dev.makeTexture(descriptor: colourDescriptor) else {
                report(label, false, "state or colour target creation failed")
                continue
            }
            var green: [Float] = [0, 1, 0, 1]
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = colour
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 1, alpha: 1)
            pass.colorAttachments[0].storeAction = .store
            if let depth {
                pass.depthAttachment.texture = depth
                pass.depthAttachment.loadAction = .clear
                pass.depthAttachment.storeAction = .dontCare
                pass.depthAttachment.clearDepth = 1.0
            }

            let commandBuffer = queue.makeCommandBuffer()!
            let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)!
            encoder.setRenderPipelineState(pipeline)
            encoder.setDepthStencilState(state)
            encoder.setStencilReferenceValue(0)
            encoder.setVertexBuffer(vertices, offset: 0, index: 0)
            encoder.setFragmentBytes(&green, length: 16, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            if let pixels = readBack(readPipe, colour, width, height) {
                let expected = shouldDraw ? pack(0, 255, 0, 255) : pack(0, 0, 255, 255)
                let ok = pixels.allSatisfy { $0 == expected }
                report(label, ok,
                       ok ? "reference zero compared against implicit stencil zero"
                          : "want=\(hex(expected)) got=\(hex(pixels[0]))")
            } else {
                refused(label)
            }
        }
    }
}
