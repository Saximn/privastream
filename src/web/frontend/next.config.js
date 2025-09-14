/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Reduce memory usage during build
    turbo: {
      memoryLimit: 2048
    }
  },
  // Optimize compilation
  swcMinify: true,
  // Reduce bundle size
  webpack: (config, { dev, isServer }) => {
    if (dev) {
      // Development optimizations
      config.optimization = {
        ...config.optimization,
        removeAvailableModules: false,
        removeEmptyChunks: false,
        splitChunks: false,
      }
    }
    
    // Optimize mediasoup-client and socket.io bundles
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    }
    
    return config
  },
  // Enable concurrent features
  reactStrictMode: true,
  // Optimize images
  images: {
    unoptimized: true
  }
}

module.exports = nextConfig