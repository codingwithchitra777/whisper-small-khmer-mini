import { NextResponse } from "next/server";

const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";

/**
 * Forward a request to the FastAPI backend and relay its JSON answer.
 * Errors come back as { error } with the backend's status, or 503 when the backend isn't running.
 */
export async function callBackend(path: string, init: RequestInit, timeoutMs: number) {
  try {
    const response = await fetch(`${FASTAPI_URL}${path}`, {
      ...init,
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      return NextResponse.json(
        { error: error.detail ?? "FastAPI request failed" },
        { status: response.status }
      );
    }

    return NextResponse.json(await response.json(), { status: response.status });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unexpected error";

    if (message.includes("ECONNREFUSED") || message.includes("fetch failed")) {
      return NextResponse.json(
        {
          error:
            "Cannot connect to the Python API server. " +
            "Please run: .\\start-api.ps1 in your project root.",
        },
        { status: 503 }
      );
    }

    return NextResponse.json({ error: message }, { status: 500 });
  }
}
