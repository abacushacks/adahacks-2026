import type { FaceBoxPayload, MetadataPayload } from "../types/protocol";


const POPUP_WIDTH = 280;
const GAP = 16;


type MetadataPopupProps = {
  face: FaceBoxPayload;
  metadata: MetadataPayload | null;
  stageHeight: number;
  stageWidth: number;
};


export function MetadataPopup({
  face,
  metadata,
  stageHeight,
  stageWidth,
}: MetadataPopupProps) {
  const scaleX = stageWidth / face.frameWidth;
  const scaleY = stageHeight / face.frameHeight;

  const boxLeft = face.x * scaleX;
  const boxTop = face.y * scaleY;
  const boxWidth = face.width * scaleX;
  const boxHeight = face.height * scaleY;

  const side =
    boxLeft + boxWidth + GAP + POPUP_WIDTH > stageWidth ? "left" : "right";

  const cardLeft =
    side === "right"
      ? boxLeft + boxWidth + GAP
      : Math.max(12, boxLeft - POPUP_WIDTH - GAP);
  const cardTop = Math.min(
    Math.max(12, boxTop),
    Math.max(12, stageHeight - 196),
  );

  return (
    <>
      <div
        className="face-box"
        style={{
          left: `${boxLeft}px`,
          top: `${boxTop}px`,
          width: `${boxWidth}px`,
          height: `${boxHeight}px`,
        }}
      />

      {metadata ? (
        <aside
          className={`metadata-popup ${side}`}
          style={{
            left: `${cardLeft}px`,
            top: `${cardTop}px`,
            width: `${POPUP_WIDTH}px`,
          }}
        >
          <p className="metadata-label">Recognized person</p>
          <h2>{metadata.name}</h2>
          <p className="relationship-line">
            {metadata.relationship || "Relationship not learned yet"}
          </p>
          <ul>
            {metadata.details.length > 0 ? (
              metadata.details.map((detail) => <li key={detail}>{detail}</li>)
            ) : (
              <li>No additional memories stored yet.</li>
            )}
          </ul>
        </aside>
      ) : (
        <div
          className="unknown-chip"
          style={{
            left: `${boxLeft}px`,
            top: `${Math.max(12, boxTop - 36)}px`,
          }}
        >
          Tracking face...
        </div>
      )}
    </>
  );
}
