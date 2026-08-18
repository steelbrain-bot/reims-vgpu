//! Backend-neutral service results used beside the submission port.

/// What an executor can prove about outstanding writes into a page window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestWriteReach {
    /// Nothing outstanding lands in any page asked about.
    Disjoint,
    /// An outstanding write lands in at least one page asked about.
    Overlap,
    /// The executor cannot name the write footprint precisely.
    Unnamed,
}

/// What a resident registry says about one target's content stamp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentContent {
    /// No resident exists under this identity.
    Absent,
    /// A resident exists but no content epoch currently vouches for it.
    Unstamped,
    /// A resident contains content from the stated semantic epoch.
    Epoch(u32),
}

/// A resident target's pixels and their physical channel order.
#[derive(Debug, Eq, PartialEq)]
pub struct TargetReadback {
    pub pixels: Vec<u8>,
    /// The bytes are BGRA8 when true and semantic RGBA8 otherwise.
    pub bgra: bool,
}

impl TargetReadback {
    /// Return semantic RGBA8, exchanging red and blue only when required.
    pub fn into_rgba8(mut self) -> Vec<u8> {
        if self.bgra {
            swap_red_blue(&mut self.pixels);
        }
        self.pixels
    }

    /// Return guest scanout order (BGRA8), exchanging only when required.
    pub fn into_bgra8(mut self) -> Vec<u8> {
        if !self.bgra {
            swap_red_blue(&mut self.pixels);
        }
        self.pixels
    }
}

fn swap_red_blue(pixels: &mut [u8]) {
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.swap(0, 2);
    }
}

#[cfg(test)]
mod tests {
    use super::TargetReadback;

    #[test]
    fn readback_converts_only_when_the_requested_order_differs() {
        let rgba = vec![1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(
            TargetReadback {
                pixels: rgba.clone(),
                bgra: false,
            }
            .into_rgba8(),
            rgba
        );
        assert_eq!(
            TargetReadback {
                pixels: rgba,
                bgra: false,
            }
            .into_bgra8(),
            vec![3, 2, 1, 4, 7, 6, 5, 8]
        );
    }
}
