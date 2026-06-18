// Client-side video blur via Insertable Streams (WebRTC §1.6 Option B).
//
// Instead of shipping every frame to the server as base64 JPEG, blurring there,
// and re-broadcasting, the host blurs faces/PII *locally* on the GPU before the
// frame enters the encoder, then produces the already-private track to the SFU.
// The Python service only returns detection coordinates; pixels travel over the
// codec, never as base64.
//
// Requires MediaStreamTrackProcessor / MediaStreamTrackGenerator (Chromium-only
// today). Viewers consume a normal WebRTC track and need no special support.

// Detection box in source-pixel coordinates: [x, y, width, height].
export type BlurBox = [number, number, number, number];

export function insertableStreamsSupported(): boolean {
  const w = globalThis as any;
  return (
    typeof w.MediaStreamTrackProcessor !== "undefined" &&
    typeof w.MediaStreamTrackGenerator !== "undefined"
  );
}

export class VideoBlurPipeline {
  // Latest detection boxes. Updated out-of-band by the detection loop; the
  // transform reads whatever is current, so blur "follows" detections without
  // blocking the frame path on the network round-trip.
  private boxes: BlurBox[] = [];
  private generator: any = null;
  private blurRadiusPx: number;

  constructor(options: { blurRadiusPx?: number } = {}) {
    this.blurRadiusPx = options.blurRadiusPx ?? 16;
  }

  /** Replace the set of regions to blur (source-pixel coordinates). */
  setBoxes(boxes: BlurBox[]): void {
    this.boxes = Array.isArray(boxes) ? boxes : [];
  }

  /**
   * Wrap an input camera track and return a new track whose frames have the
   * current boxes blurred. Produce the returned track to the SFU.
   */
  start(inputTrack: MediaStreamTrack): MediaStreamTrack {
    const w = globalThis as any;
    if (!insertableStreamsSupported()) {
      throw new Error(
        "Insertable Streams (MediaStreamTrackProcessor/Generator) not supported in this browser — use a Chromium-based browser."
      );
    }

    const processor = new w.MediaStreamTrackProcessor({ track: inputTrack });
    const generator = new w.MediaStreamTrackGenerator({ kind: "video" });
    this.generator = generator;

    let canvas: OffscreenCanvas | null = null;
    let ctx: OffscreenCanvasRenderingContext2D | null = null;

    const transformer = new TransformStream<VideoFrame, VideoFrame>({
      transform: (frame, controller) => {
        const width = frame.displayWidth;
        const height = frame.displayHeight;

        if (!canvas || canvas.width !== width || canvas.height !== height) {
          canvas = new OffscreenCanvas(width, height);
          ctx = canvas.getContext("2d");
        }
        const c = ctx!;

        // Paint the sharp frame, then re-paint each region through a blur
        // filter clipped to that region.
        c.filter = "none";
        c.drawImage(frame, 0, 0, width, height);

        const boxes = this.boxes;
        if (boxes.length > 0) {
          c.filter = `blur(${this.blurRadiusPx}px)`;
          for (const box of boxes) {
            const [x, y, bw, bh] = box;
            if (!bw || !bh) continue;
            c.save();
            c.beginPath();
            c.rect(x, y, bw, bh);
            c.clip();
            c.drawImage(frame, 0, 0, width, height);
            c.restore();
          }
          c.filter = "none";
        }

        const out = new VideoFrame(canvas as unknown as CanvasImageSource, {
          timestamp: frame.timestamp ?? 0,
        });
        frame.close();
        controller.enqueue(out);
      },
    });

    // Drive the pipeline. Errors (e.g. track ended) are logged, not thrown.
    processor.readable
      .pipeThrough(transformer)
      .pipeTo(generator.writable)
      .catch((err: unknown) => {
        console.error("[BlurPipeline] pipeline terminated:", err);
      });

    return generator as MediaStreamTrack;
  }

  stop(): void {
    try {
      this.generator?.stop?.();
    } catch {
      /* already stopped */
    }
    this.generator = null;
    this.boxes = [];
  }
}
