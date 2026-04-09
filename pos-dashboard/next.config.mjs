/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['recharts', 'lucide-react', 'framer-motion'],
  allowedDevOrigins: ['127.0.2.2', 'localhost']
};

export default nextConfig;
