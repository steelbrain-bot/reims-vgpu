//! Framing and wire decoders (batch B).

pub mod blit;
pub mod compute;
/// Cross-checks between the closure ledger and these decoders.
#[cfg(test)]
mod ledger;
pub mod render;
pub mod resource;
pub mod stream;
