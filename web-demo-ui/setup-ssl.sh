#!/bin/bash

# SSL Certificate Setup Script
# This script creates self-signed certificates for development or helps set up production certificates

echo "🔒 SSL Certificate Setup"
echo "========================"

# Create ssl directory if it doesn't exist
mkdir -p ssl
mkdir -p backend/ssl
mkdir -p mediasoup-server/ssl
mkdir -p frontend/ssl

echo "Choose SSL setup option:"
echo "1) Generate self-signed certificates (development)"
echo "2) Use existing certificates (production)"
echo "3) Use Let's Encrypt (production - requires domain)"

read -p "Enter your choice (1-3): " choice

case $choice in
  1)
    echo "🔧 Generating self-signed certificates for development..."
    
    # Generate private key
    openssl genpkey -algorithm RSA -out ssl/key.pem -pkcs8 -pass pass:
    
    # Generate certificate signing request
    openssl req -new -key ssl/key.pem -out ssl/csr.pem -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    
    # Generate self-signed certificate
    openssl x509 -req -in ssl/csr.pem -signkey ssl/key.pem -out ssl/cert.pem -days 365
    
    # Copy to all service directories
    cp ssl/cert.pem backend/ssl/
    cp ssl/key.pem backend/ssl/
    cp ssl/cert.pem mediasoup-server/ssl/
    cp ssl/key.pem mediasoup-server/ssl/
    cp ssl/cert.pem frontend/ssl/
    cp ssl/key.pem frontend/ssl/
    
    # Clean up CSR
    rm ssl/csr.pem
    
    echo "✅ Self-signed certificates generated successfully!"
    echo "   Certificate: ssl/cert.pem"
    echo "   Private Key: ssl/key.pem"
    echo ""
    echo "⚠️  WARNING: Self-signed certificates will show security warnings in browsers."
    echo "   For production, use option 3 (Let's Encrypt) or provide real certificates."
    ;;
    
  2)
    echo "📋 To use existing certificates:"
    echo "1. Place your certificate file at: ssl/cert.pem"
    echo "2. Place your private key file at: ssl/key.pem"
    echo "3. Make sure they have proper permissions (600 for key.pem)"
    echo "4. Run this script again to copy them to service directories"
    
    if [[ -f "ssl/cert.pem" && -f "ssl/key.pem" ]]; then
      # Copy to all service directories
      cp ssl/cert.pem backend/ssl/
      cp ssl/key.pem backend/ssl/
      cp ssl/cert.pem mediasoup-server/ssl/
      cp ssl/key.pem mediasoup-server/ssl/
      cp ssl/cert.pem frontend/ssl/
      cp ssl/key.pem frontend/ssl/
      
      # Set proper permissions
      chmod 600 ssl/key.pem backend/ssl/key.pem mediasoup-server/ssl/key.pem frontend/ssl/key.pem
      chmod 644 ssl/cert.pem backend/ssl/cert.pem mediasoup-server/ssl/cert.pem frontend/ssl/cert.pem
      
      echo "✅ Existing certificates copied to all service directories!"
    else
      echo "❌ Certificate files not found. Please place them in ssl/ directory first."
    fi
    ;;
    
  3)
    echo "🌍 Let's Encrypt Setup"
    echo "This requires:"
    echo "- A registered domain name"
    echo "- Domain pointing to this server's IP"
    echo "- Port 80 and 443 open for verification"
    echo ""
    
    read -p "Enter your domain name: " domain
    
    if command -v certbot &> /dev/null; then
      echo "🔧 Running certbot for domain: $domain"
      sudo certbot certonly --standalone -d $domain
      
      # Copy Let's Encrypt certificates
      sudo cp /etc/letsencrypt/live/$domain/fullchain.pem ssl/cert.pem
      sudo cp /etc/letsencrypt/live/$domain/privkey.pem ssl/key.pem
      
      # Copy to all service directories
      cp ssl/cert.pem backend/ssl/
      cp ssl/key.pem backend/ssl/
      cp ssl/cert.pem mediasoup-server/ssl/
      cp ssl/key.pem mediasoup-server/ssl/
      cp ssl/cert.pem frontend/ssl/
      cp ssl/key.pem frontend/ssl/
      
      # Fix permissions
      sudo chown $USER:$USER ssl/cert.pem ssl/key.pem
      chmod 644 ssl/cert.pem
      chmod 600 ssl/key.pem
      
      echo "✅ Let's Encrypt certificates installed successfully!"
      echo ""
      echo "📝 Don't forget to:"
      echo "1. Set up certificate auto-renewal: sudo crontab -e"
      echo "2. Add: 0 12 * * * /usr/bin/certbot renew --quiet"
      echo "3. Update your environment variables with your domain"
    else
      echo "❌ certbot not found. Please install certbot first:"
      echo "   Ubuntu/Debian: sudo apt-get install certbot"
      echo "   CentOS/RHEL: sudo yum install certbot"
      echo "   macOS: brew install certbot"
    fi
    ;;
    
  *)
    echo "❌ Invalid choice. Exiting."
    exit 1
    ;;
esac

echo ""
echo "🔧 Next steps:"
echo "1. Update your .env files with SSL_ENABLED=true"
echo "2. Update service URLs to use https:// instead of http://"
echo "3. Restart all services"
echo "4. Test HTTPS access in your browser"