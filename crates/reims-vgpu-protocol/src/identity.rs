//! Non-interchangeable identities and quantities used by the semantic core.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

macro_rules! scalar_newtype {
    ($name:ident, $inner:ty) => {
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($inner);

        impl $name {
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl From<$inner> for $name {
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

scalar_newtype!(TaskId, u32);
scalar_newtype!(ResourceNamespaceId, u32);
scalar_newtype!(MappingId, u32);
scalar_newtype!(SurfaceId, u32);
scalar_newtype!(SurfaceBackingId, u64);
scalar_newtype!(StorageId, u64);
scalar_newtype!(GuestVirtualAddress, u64);
scalar_newtype!(GuestPhysicalAddress, u64);
scalar_newtype!(ByteOffset, u64);
scalar_newtype!(ByteLength, u64);
scalar_newtype!(SubmissionId, u64);
scalar_newtype!(BackingGeneration, u64);
scalar_newtype!(ContentVersion, u64);
scalar_newtype!(PlaneIndex, u32);

impl fmt::LowerHex for GuestVirtualAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl fmt::LowerHex for GuestPhysicalAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

/// A task-local slot in a typed resource namespace.
#[repr(transparent)]
pub struct ObjectRef<T> {
    value: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> ObjectRef<T> {
    pub const fn new(value: u32) -> Self {
        Self {
            value,
            marker: PhantomData,
        }
    }

    pub const fn get(self) -> u32 {
        self.value
    }
}

impl<T> Clone for ObjectRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ObjectRef<T> {}

impl<T> fmt::Debug for ObjectRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ObjectRef").field(&self.value).finish()
    }
}

impl<T> PartialEq for ObjectRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> Eq for ObjectRef<T> {}

impl<T> PartialOrd for ObjectRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ObjectRef<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T> Hash for ObjectRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

/// A generational internal identity for one typed resource lifetime.
pub struct ResourceId<T> {
    index: u32,
    generation: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> ResourceId<T> {
    pub const fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            marker: PhantomData,
        }
    }

    pub const fn index(self) -> u32 {
        self.index
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl<T> Clone for ResourceId<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ResourceId<T> {}

impl<T> fmt::Debug for ResourceId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceId")
            .field("index", &self.index)
            .field("generation", &self.generation)
            .finish()
    }
}

impl<T> PartialEq for ResourceId<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.index, self.generation) == (other.index, other.generation)
    }
}

impl<T> Eq for ResourceId<T> {}

impl<T> PartialOrd for ResourceId<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ResourceId<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.index, self.generation).cmp(&(other.index, other.generation))
    }
}

impl<T> Hash for ResourceId<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum Buffer {}
    enum Texture {}

    #[test]
    fn typed_namespaces_do_not_share_a_runtime_representation_owner() {
        let buffer = ObjectRef::<Buffer>::new(7);
        let texture = ObjectRef::<Texture>::new(7);
        assert_eq!(buffer.get(), texture.get());

        let first = ResourceId::<Buffer>::new(3, 4);
        let reused = ResourceId::<Buffer>::new(3, 5);
        assert_ne!(first, reused);
    }
}
