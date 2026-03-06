#!/bin/bash
# EA-NN Container Management Script
# Replaces the virtual environment setup with containerized deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"  # Script is in project root

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}🐳 EA-NN Container Management${NC}"
    echo -e "${BLUE}===============================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_requirements() {
    print_header
    echo "🔍 Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker Desktop."
        exit 1
    fi
    print_success "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose found"
    
    # Check NVIDIA Docker (for GPU) - more lenient check
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        
        # Test basic NVIDIA Docker support
        if docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
            print_success "NVIDIA Docker runtime configured correctly"
            GPU_AVAILABLE=true
        elif docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
            print_success "NVIDIA Docker support detected (legacy mode)"
            GPU_AVAILABLE=true
        else
            print_warning "NVIDIA Docker runtime not configured, will use CPU-only mode"
            print_warning "To enable GPU: install nvidia-docker2 or Docker Desktop with GPU support"
            GPU_AVAILABLE=false
        fi
    else
        print_warning "NVIDIA GPU not detected, will use CPU-only mode"
        GPU_AVAILABLE=false
    fi
    
    echo ""
}

show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  dev         Start development environment with DevContainer"
    echo "  prod        Start production deployment"
    echo "  build       Build container images"
    echo "  stop        Stop all containers"
    echo "  clean       Clean up containers and images"
    echo "  logs        Show container logs"
    echo "  shell       Open shell in development container"
    echo "  monitor     Start GPU monitoring"
    echo "  scale       Scale production deployment"
    echo ""
    echo "Options:"
    echo "  --no-gpu    Disable GPU support"
    echo "  --rebuild   Force rebuild of images"
    echo "  --detach    Run in background"
    echo ""
    echo "Examples:"
    echo "  $0 dev              # Start development environment"
    echo "  $0 prod --detach    # Start production in background"
    echo "  $0 build --rebuild  # Force rebuild all images"
    echo "  $0 scale 3          # Scale to 3 production instances"
}

build_images() {
    local rebuild=${1:-false}
    
    echo "🏗️ Building EA-NN container images..."
    
    cd "$PROJECT_ROOT"
    
    # Prefer simple compose file for better compatibility
    local compose_file="docker-compose-simple.yml"
    
    echo "📄 Using compose file: $compose_file"
    
    if [ "$rebuild" = true ]; then
        echo "🔄 Force rebuilding images..."
        docker-compose -f "$compose_file" build --no-cache
    else
        docker-compose -f "$compose_file" build
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Images built successfully"
    else
        print_error "Image build failed"
        exit 1
    fi
}

start_development() {
    local detach=${1:-false}
    
    echo "🚀 Starting EA-NN development environment..."
    
    cd "$PROJECT_ROOT"
    
    # Use simple compose file for better compatibility
    local compose_file="docker-compose-simple.yml"
    
    if [ "$GPU_AVAILABLE" = true ] && [ "$NO_GPU" != true ]; then
        print_success "Starting with GPU acceleration"
    else
        print_warning "Starting in CPU-only mode (GPU will be auto-detected in container)"
    fi
    
    if [ "$detach" = true ]; then
        docker-compose -f "$compose_file" --profile development up -d
    else
        echo "📝 Note: Press Ctrl+C to stop the development environment"
        echo "🔗 Access will be available at:"
        echo "   📱 Web Interface: http://localhost:5000"
        echo "   📊 Jupyter Lab: http://localhost:8888"
        echo ""
        docker-compose -f "$compose_file" --profile development up
    fi
}

start_production() {
    local detach=${1:-true}
    local profile="production"
    
    echo "🚀 Starting EA-NN production deployment..."
    
    cd "$PROJECT_ROOT"
    
    # Choose profile based on GPU availability
    if [ "$GPU_AVAILABLE" = true ] && [ "$NO_GPU" != true ]; then
        print_success "Starting with GPU acceleration"
        profile="production"
    else
        print_warning "Starting in CPU-only mode"
        profile="production-cpu"
    fi
    
    if [ "$detach" = true ]; then
        docker-compose --profile "$profile" up -d
    else
        docker-compose --profile "$profile" up
    fi
    
    echo ""
    print_success "Production deployment started!"
    echo "🌐 Access the simulation at: http://localhost"
    echo "📊 Monitor with: docker logs ea-nn-simulation"
    if [ "$GPU_AVAILABLE" = true ]; then
        echo "🎮 Check GPU usage: docker exec ea-nn-simulation nvidia-smi"
    fi
}

stop_containers() {
    echo "🛑 Stopping EA-NN containers..."
    
    cd "$PROJECT_ROOT"
    docker-compose down
    
    print_success "All containers stopped"
}

clean_up() {
    echo "🧹 Cleaning up EA-NN containers and images..."
    
    cd "$PROJECT_ROOT"
    docker-compose down --rmi all --volumes --remove-orphans
    
    # Clean up unused images
    docker image prune -f
    
    print_success "Cleanup complete"
}

show_logs() {
    local service=${1:-""}
    
    cd "$PROJECT_ROOT"
    
    if [ -n "$service" ]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

open_shell() {
    echo "🐚 Opening shell in development container..."
    
    cd "$PROJECT_ROOT"
    
    # Try to connect to running dev container
    if docker-compose ps | grep ea-nn-dev | grep Up > /dev/null; then
        docker-compose exec ea-nn-dev /bin/bash
    else
        echo "Starting development container..."
        docker-compose --profile development run --rm ea-nn-dev /bin/bash
    fi
}

start_monitoring() {
    echo "📊 Starting GPU monitoring..."
    
    cd "$PROJECT_ROOT"
    docker-compose --profile monitoring up -d
    
    print_success "GPU monitoring started on http://localhost:9400"
}

scale_production() {
    local replicas=${1:-1}
    
    echo "⚖️ Scaling production to $replicas instances..."
    
    cd "$PROJECT_ROOT"
    docker-compose --profile production up -d --scale ea-nn-prod="$replicas"
    
    print_success "Scaled to $replicas production instances"
}

# Parse arguments
COMMAND=""
REBUILD=false
DETACH=false
NO_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        dev|prod|build|stop|clean|logs|shell|monitor|scale)
            COMMAND="$1"
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        --no-gpu)
            NO_GPU=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            if [ "$COMMAND" = "scale" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
                SCALE_COUNT="$1"
            elif [ "$COMMAND" = "logs" ]; then
                LOG_SERVICE="$1"
            else
                echo "Unknown option: $1"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Main execution
case $COMMAND in
    "")
        show_usage
        ;;
    "dev")
        check_requirements
        build_images $REBUILD
        start_development $DETACH
        ;;
    "prod")
        check_requirements
        build_images $REBUILD
        start_production $DETACH
        ;;
    "build")
        check_requirements
        build_images $REBUILD
        ;;
    "stop")
        stop_containers
        ;;
    "clean")
        clean_up
        ;;
    "logs")
        show_logs "$LOG_SERVICE"
        ;;
    "shell")
        open_shell
        ;;
    "monitor")
        check_requirements
        start_monitoring
        ;;
    "scale")
        scale_production "${SCALE_COUNT:-1}"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
