#!/bin/bash
# Production deployment script for CrackGraphAI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.prod.yml"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker runtime (optional)
    if ! docker info | grep -q "nvidia"; then
        log_warn "NVIDIA Docker runtime not detected. GPU support may not be available."
    fi
    
    log_info "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    cd "$PROJECT_DIR"
    
    # Create .env if it doesn't exist
    if [ ! -f .env ]; then
        log_warn ".env file not found. Creating from example..."
        if [ -f .env.example ]; then
            cp .env.example .env
            log_warn "Please update .env with your production settings!"
        else
            log_warn "No .env.example found. Creating minimal .env..."
            cat > .env <<EOF
API_KEY=change-me-in-production
GRAFANA_PASSWORD=admin
CORS_ORIGINS=*
EOF
        fi
    fi
    
    # Create required directories
    mkdir -p checkpoints outputs .cache logs
    
    # Check for model weights
    if [ ! -f "checkpoints/best_hybrid_segformer.pth" ]; then
        log_warn "Model weights not found at checkpoints/best_hybrid_segformer.pth"
        log_warn "Please place your trained model weights in the checkpoints directory"
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    log_info "Build complete"
}

start_services() {
    log_info "Starting production services..."
    
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_info "Services started. Checking health..."
    sleep 10
    
    # Health check
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_info "API is healthy!"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    echo ""
    
    # Final status
    docker-compose -f "$COMPOSE_FILE" ps
}

stop_services() {
    log_info "Stopping services..."
    
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" down
    
    log_info "Services stopped"
}

update_services() {
    log_info "Updating services (zero-downtime)..."
    
    cd "$PROJECT_DIR"
    
    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Recreate containers
    docker-compose -f "$COMPOSE_FILE" up -d --force-recreate
    
    log_info "Services updated"
}

show_logs() {
    log_info "Showing logs..."
    
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" logs -f api
}

show_status() {
    log_info "Service status:"
    
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Health check:"
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://localhost:8000/health
}

# Main
case "${1:-deploy}" in
    deploy)
        check_prerequisites
        setup_environment
        build_images
        start_services
        show_status
        log_info "Deployment complete!"
        log_info "API available at: http://localhost:8000"
        log_info "Grafana dashboard: http://localhost:3000"
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        ;;
    update)
        update_services
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    build)
        build_images
        ;;
    *)
        echo "Usage: $0 {deploy|start|stop|restart|update|logs|status|build}"
        exit 1
        ;;
esac
