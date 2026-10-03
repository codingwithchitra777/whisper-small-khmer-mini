import { NextRequest } from "next/server";
import { callBackend } from "../../../../lib/backend";

// Poll a job: status, progress in chunks, and the segments once it is done.
export async function GET(_request: NextRequest, { params }: { params: { id: string } }) {
  return callBackend(`/jobs/${encodeURIComponent(params.id)}`, { method: "GET" }, 30_000);
}
