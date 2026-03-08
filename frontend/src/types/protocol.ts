export type ConnectionState = "idle" | "connecting" | "connected";


export type FaceBoxPayload = {
  x: number;
  y: number;
  width: number;
  height: number;
  frameWidth: number;
  frameHeight: number;
};


export type MetadataPayload = {
  personId: number;
  name: string;
  relationship: string;
  details: string[];
};


export type TrackingUpdateMessage = {
  type: "tracking_update";
  trackingStatus: "recognized" | "unknown";
  face: FaceBoxPayload;
  metadata: MetadataPayload | null;
};


export type TrackingLostMessage = {
  type: "tracking_lost";
};


export type ErrorMessage = {
  type: "error";
  message: string;
};


export type ControlMessage = {
  type:
    | "session_ready"
    | "session_ack"
    | "session_stopped"
    | "speaker_reference_ready"
    | "profile_learned";
  personId?: number;
  name?: string;
};


export type ServerMessage =
  | TrackingUpdateMessage
  | TrackingLostMessage
  | ErrorMessage
  | ControlMessage;
