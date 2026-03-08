function int16ToBase64(samples: Int16Array): string {
  const bytes = new Uint8Array(samples.buffer);
  let binary = "";
  const chunkSize = 0x8000;

  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }

  return window.btoa(binary);
}


export function float32ToBase64Pcm(chunk: Float32Array): string {
  const pcm = new Int16Array(chunk.length);
  for (let index = 0; index < chunk.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, chunk[index]));
    pcm[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }
  return int16ToBase64(pcm);
}


export function mergeFloat32Chunks(chunks: Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);

  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });

  return merged;
}
