export function captureVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): string | null {
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    return null;
  }

  const targetWidth = 640;
  const targetHeight = Math.round((video.videoHeight / video.videoWidth) * targetWidth);

  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }

  context.drawImage(video, 0, 0, targetWidth, targetHeight);
  return canvas.toDataURL("image/jpeg", 0.72).split(",")[1] ?? null;
}
