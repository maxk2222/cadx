import * as THREE from "three";

export type MeshTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array;

/**
 * Serialized copy of a vertex attribute — the raw backing array plus the
 * layout info needed to rebuild an attribute that reads IDENTICALLY via
 * getX/getY/getZ off-thread (interleaved strides, normalized integer
 * formats). The array is always a private copy: transfer it to a worker
 * freely — the rendered geometry keeps its own buffer.
 */
export interface PositionsWire {
  array: MeshTypedArray;
  itemSize: number;
  normalized: boolean;
  /** set when the source attribute is interleaved */
  interleaved: { stride: number; offset: number } | null;
  count: number;
}

/** copy a position attribute into a transferable wire payload. */
export function positionsToWire(
  attr: THREE.BufferAttribute | THREE.InterleavedBufferAttribute | undefined
): PositionsWire | null {
  if (!attr) return null;
  const il = attr as THREE.InterleavedBufferAttribute;
  if (il.isInterleavedBufferAttribute) {
    return {
      array: il.data.array.slice() as MeshTypedArray,
      itemSize: il.itemSize,
      normalized: il.normalized,
      interleaved: { stride: il.data.stride, offset: il.offset },
      count: il.count,
    };
  }
  const b = attr as THREE.BufferAttribute;
  return {
    array: b.array.slice() as MeshTypedArray,
    itemSize: b.itemSize,
    normalized: b.normalized,
    interleaved: null,
    count: b.count,
  };
}

/** copy an index attribute into a transferable wire payload. */
export function indexToWire(
  index: THREE.BufferAttribute | null
): MeshTypedArray | null {
  return index ? (index.array.slice() as MeshTypedArray) : null;
}

/** rebuild an attribute that reads exactly like the original. */
export function positionsAttribute(
  p: PositionsWire
): THREE.BufferAttribute | THREE.InterleavedBufferAttribute {
  if (p.interleaved) {
    return new THREE.InterleavedBufferAttribute(
      new THREE.InterleavedBuffer(p.array, p.interleaved.stride),
      p.itemSize,
      p.interleaved.offset,
      p.normalized
    );
  }
  return new THREE.BufferAttribute(p.array, p.itemSize, p.normalized);
}
