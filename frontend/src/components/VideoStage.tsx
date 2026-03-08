import { RefObject, useEffect, useState } from "react";

import type { TrackingUpdateMessage } from "../types/protocol";
import { MetadataPopup } from "./MetadataPopup";


type VideoStageProps = {
  tracking: TrackingUpdateMessage | null;
  videoRef: RefObject<HTMLVideoElement>;
};


export function VideoStage({ tracking, videoRef }: VideoStageProps) {
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const node = videoRef.current;
    if (!node) {
      return undefined;
    }

    const syncSize = () => {
      setStageSize({
        width: node.clientWidth,
        height: node.clientHeight,
      });
    };

    syncSize();

    const resizeObserver = new ResizeObserver(syncSize);
    resizeObserver.observe(node);
    node.addEventListener("loadedmetadata", syncSize);

    return () => {
      resizeObserver.disconnect();
      node.removeEventListener("loadedmetadata", syncSize);
    };
  }, [videoRef]);

  const trackedFace = tracking?.face ?? null;
  const recognizedMetadata =
    tracking?.trackingStatus === "recognized" ? tracking.metadata : null;

  return (
    <section className="stage-shell">
      <div className="video-stage">
        <video
          autoPlay
          className="live-video"
          muted
          playsInline
          ref={videoRef}
        />

        {trackedFace &&
        stageSize.width > 0 &&
        stageSize.height > 0 ? (
          <MetadataPopup
            face={trackedFace}
            metadata={recognizedMetadata}
            stageHeight={stageSize.height}
            stageWidth={stageSize.width}
          />
        ) : null}
      </div>
    </section>
  );
}
