/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow fetching from local FastAPI backend
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
