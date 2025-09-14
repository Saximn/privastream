const API_CONFIG = {
  BACKEND_URL: process.env.BACKEND_URL || "http://localhost:5000",
  SFU_URL: process.env.SFU_URL || "http://localhost:3001/",
  VIDEO_API_URL: process.env.VIDEO_API_URL || "http://localhost:5001/",
  AUDIO_API_URL: process.env.AUDIO_API_URL || "http://localhost:5002/"
};

export default API_CONFIG;
