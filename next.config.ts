import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,

  // Use an empty turbopack block to allow custom webpack configuration to run.
  turbopack: {},

  // Use the traditional webpack configuration to reliably mark external dependencies.
  webpack: (config, { isServer }) => {
    // We only need this on the server side (Node.js runtime)
    if (isServer) {
      // Initialize externals array if it doesn't exist
      if (!config.externals) {
        config.externals = [];
      }
      
      // Mark these modules as external to prevent Next.js/Webpack from trying to bundle them.
      // 1. The optional dependencies causing the "Module not found" error:
      config.externals.push(
        'cohere-ai',
        'voyageai',
      );
      
      // 2. Add 'chromadb-client' itself as an external module to ensure it's not processed
      // by the internal bundler, which helps resolve complex import chains.
      config.externals.push('chromadb-client');

      config.externals.push('pdf-parse');
    }

    return config;
  },
  
  // The 'experimental' block was removed to avoid potential conflicts with Turbopack/Webpack.
};

export default nextConfig;