//! Self-contained Vulkan execution backend.
//!
//! All host GPU work lives under `backend/vulkan/`, driven by `ash`. Product draw encode uses the
//! internal [`engine`] (persistent ash context + content-keyed caches). This
//! crate has no external graphics-executor dependency; AIR translation comes
//! from the pinned public `metal2vulkan` crate.
//!
//! Product submission and guest-lifetime reset enter through the device-owned
//! [`crate::runtime::executor::Executor`] implementation.
//!
//! [`caps`] classifies the bound host GPU into the four-cell support matrix
//! (unified/discrete memory × has/has-no import) that every path here must keep
//! working. [`policy`] owns the independent unified and discrete placement and
//! scheduling choices; host-pointer import remains an orthogonal capability.
//!
//! [`translate`] is the matching seam for *state*: decoded Metal formats and
//! pipeline enums become Vulkan ones there and nowhere else, so the same
//! decision cannot be made twice with two different answers.

pub mod caps;
pub mod engine;
pub mod policy;
pub mod translate;
