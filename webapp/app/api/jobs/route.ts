import { NextRequest } from "next/server";
import { callBackend } from "../../../lib/backend";

// Queue a YouTube URL; answers at once with { job_id, status, ... }.
export async function POST(request: NextRequest) {
  const body = await request.json();
  return callBackend(
    "/jobs",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    30_000
  );
}
